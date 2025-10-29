"""
model_train.py
---------------
Monthly model retraining script for Credit Default prediction.
Triggered via Airflow DAG once sufficient data are available.

Steps:
1. Load configuration (ML_config.yaml)
2. Identify available snapshots and build rolling train/val/test/oot splits
3. Load & merge feature_store and label_store
4. Preprocess: impute, encode categorical vars, scale (optional)
5. Train XGBoost with randomized search
6. Evaluate (AUC, Gini, etc.)
7. Extract feature importance for PSI monitoring
8. Save model and metadata under model_store/candidate_models/<snapshot_date>/
"""

import os
import sys
import yaml
import json
import joblib
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, make_scorer

import xgboost as xgb
import contextlib, io, logging
import warnings
from xgboost import XGBClassifier
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Utility: resolve paths relative to project root
# -------------------------------------------------------------------------
def resolve_relative_path(path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    return os.path.abspath(os.path.join(base_dir, path))

# -------------------------------------------------------------------------
# Utility: load YAML config
# -------------------------------------------------------------------------
def load_config(config_path="config/ML_config.yaml"):
    abs_path = resolve_relative_path(config_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    with open(abs_path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------------------------------------------------
# Step 1 – load and merge feature + label data for relevant months
# -------------------------------------------------------------------------
"""
Important note: snapshot_date on feature and label stores relates to date of creation. Since labels are
created on a 4 month lag due to 30DPD_4MOB logic, i.e. labels relating to loans with loan_start_date = 2023-01-01
are captured in label_store/2023-05-01.parquet, hence lag_months = 4 is used to align feature and labels
"""
def load_datasets(gold_dir, lag_months=4, min_months_required=9, max_snapshot_date=None):
    feat_dir = os.path.join(gold_dir, "feature_store")
    lab_dir = os.path.join(gold_dir, "label_store")

    feature_files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".parquet")])
    label_files = sorted([f for f in os.listdir(lab_dir) if f.endswith(".parquet")])

    # Convert filenames to datetime for easier offset logic
    feat_dates = [datetime.strptime(f.replace(".parquet", "").replace("_", "-"), "%Y-%m-%d") for f in feature_files]
    lab_dates = [datetime.strptime(f.replace(".parquet", "").replace("_", "-"), "%Y-%m-%d") for f in label_files]

    # Cutoff to prevent using "future" data
    if max_snapshot_date:
        feat_dates = [d for d in feat_dates if d <= max_snapshot_date]
        lab_dates = [d for d in lab_dates if d <= max_snapshot_date]
    
    aligned_pairs = []
    for fd in feat_dates:
        expected_label_date = fd + relativedelta(months=lag_months)
        if expected_label_date in lab_dates:
            aligned_pairs.append((fd, expected_label_date))

    if len(aligned_pairs) < min_months_required:
        logger.info(f"⏭️ Only {len(aligned_pairs)} aligned months (need {min_months_required}). Skipping retrain.")
        return None, None

    logger.info(f"📂 Found {len(aligned_pairs)} aligned feature/label pairs.")
    
    feature_dfs = []
    for f_date, l_date in aligned_pairs[-min_months_required:]:
        f_str = f_date.strftime("%Y_%m_%d")
        l_str = l_date.strftime("%Y_%m_%d")

        feat_path = os.path.join(feat_dir, f"{f_str}.parquet")
        lab_path = os.path.join(lab_dir, f"{l_str}.parquet")

        feat_df = pd.read_parquet(feat_path)
        lab_df = pd.read_parquet(lab_path)
        df = feat_df.merge(lab_df, on=["Customer_ID", "loan_start_date"], how="inner")

        df["snapshot_tag"] = f_str  # snapshot corresponds to feature month
        feature_dfs.append(df)

    full = pd.concat(feature_dfs, ignore_index=True)
    latest_feature_snapshot = aligned_pairs[-1][0].strftime("%Y-%m-%d")
    logger.info(f"✅ Loaded {len(full)} rows ({aligned_pairs[0][0].date()} → {aligned_pairs[-1][0].date()})")
    return full, latest_feature_snapshot

# -------------------------------------------------------------------------
# Step 2 – build rolling time splits
# -------------------------------------------------------------------------
def make_splits(df, snapshot_dates, split_cfg):
    """
    Splits data chronologically into train / val / test / oot sets
    according to split configuration in ML_config.yaml.
    """
    sorted_dates = sorted(snapshot_dates)
    total_needed = (
        split_cfg["train_months"] +
        split_cfg["val_months"] +
        split_cfg["test_months"] +
        split_cfg["oot_months"]
    )

    if len(sorted_dates) < total_needed:
        logger.warning(
            f"⏭️ Not enough months ({len(sorted_dates)} available, need {total_needed}). Skipping retrain."
        )
        return None, None, None, None

    idx_train_end = split_cfg["train_months"]
    idx_val_end = idx_train_end + split_cfg["val_months"]
    idx_test_end = idx_val_end + split_cfg["test_months"]
    idx_oot_end = idx_test_end + split_cfg["oot_months"]

    # chronological slicing
    train = df[df["snapshot_tag"].isin(sorted_dates[:idx_train_end])]
    val = df[df["snapshot_tag"].isin(sorted_dates[idx_train_end:idx_val_end])]
    test = df[df["snapshot_tag"].isin(sorted_dates[idx_val_end:idx_test_end])]
    oot = df[df["snapshot_tag"].isin(sorted_dates[idx_test_end:idx_oot_end])]

    logger.info(f"📊 Split summary: train={len(train)}, val={len(val)}, test={len(test)}, oot={len(oot)}")
    return train, val, test, oot

# -------------------------------------------------------------------------
# Step 3 – preprocessing pipeline
# -------------------------------------------------------------------------
def build_preprocessor(config, X_train):
    cat_feats = config["categorical_features"]
    num_impute_rules = config["imputation"]["numeric"].copy()
    cat_impute_rules = config["imputation"]["categorical"].copy()
    scale_cfg = config["scaling"]

    # detect all numeric & categorical columns
    all_numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
    all_categorical = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    auto_handled = {"numeric": [], "categorical": []}
    
    # Check numeric columns
    for c in all_numeric:
        if c not in num_impute_rules and X_train[c].isna().any():
            num_impute_rules[c] = 0
            auto_handled["numeric"].append(c)
    
    # Check categorical columns
    for c in all_categorical:
        if (c not in cat_impute_rules and c not in cat_feats) and X_train[c].isna().any():
            cat_impute_rules[c] = "Unknown"
            auto_handled["categorical"].append(c)

    # build feature lists
    num_feats = list(num_impute_rules.keys())
    cat_feats = list(set(cat_feats + list(cat_impute_rules.keys())))

    # imputers
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # encoder
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # scaler (optional)
    scaler = StandardScaler() if scale_cfg["apply"] else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", num_imputer), ("scaler", scaler)]), num_feats),
            ("cat", Pipeline(steps=[("imputer", cat_imputer), ("encoder", encoder)]), cat_feats)
        ],
        remainder="passthrough"
    )

    # return both preprocessor and metadata
    return preprocessor, auto_handled

# -------------------------------------------------------------------------
# Step 4 – train model
# -------------------------------------------------------------------------
def train_xgb_model(X_train, y_train, config):
    import contextlib, io, logging, warnings, numpy as np
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    from sklearn.metrics import roc_auc_score

    warnings.filterwarnings("ignore", message=".*use_label_encoder.*")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    model_cfg = config["model"]
    param_grid = model_cfg["param_grid"]

    # --- Base model (no deprecated params) ---
    xgb_clf = xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=88
    )

    # --- Use built-in scorer to avoid "needs_proba" error ---
    auc_scorer = "roc_auc"

    # --- Configure search ---
    search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_grid,
        scoring=auc_scorer,
        n_iter=model_cfg["n_iter"],
        cv=StratifiedKFold(
            n_splits=model_cfg["cv_folds"], shuffle=True, random_state=42
        ),
        verbose=0,
        random_state=42,
        n_jobs=-1,
        error_score=np.nan
    )

    # --- Silence XGBoost logs ---
    xgb.set_config(verbosity=0)
    logging.getLogger("xgboost").setLevel(logging.CRITICAL)
    logging.getLogger("sklearn").setLevel(logging.ERROR)

    # --- Run search safely ---
    with contextlib.redirect_stderr(io.StringIO()):
        search.fit(X_train, y_train)

    # --- Handle NaN CV AUC fallback ---
    if np.isnan(search.best_score_):
        logger.warning("⚠️ CV AUC returned NaN — recalculating AUC on full training set.")
        best_model = search.best_estimator_
        y_pred_train = best_model.predict_proba(X_train)[:, 1]
        cv_auc = roc_auc_score(y_train, y_pred_train)
    else:
        cv_auc = search.best_score_

    logger.info(f"🏆 Best params={search.best_params_}, CV AUC={cv_auc:.4f}")
    return search.best_estimator_, search.best_params_, cv_auc
# -------------------------------------------------------------------------
# Step 5 – evaluate model
# -------------------------------------------------------------------------
def evaluate_model(model, X, y, metrics_to_compute):
    """
    Evaluate model performance for the specified metrics.
    Probabilities are used for AUC, Gini, KS; 0.5 threshold for classification metrics.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    results = {}
    for metric in metrics_to_compute:
        if metric == "auc":
            results["auc"] = roc_auc_score(y, y_prob)
        elif metric == "gini":
            results["gini"] = 2 * roc_auc_score(y, y_prob) - 1
        elif metric == "ks":
            results["ks"] = ks_2samp(y_prob[y == 1], y_prob[y == 0]).statistic
        elif metric == "f1":
            results["f1"] = f1_score(y, y_pred)
        elif metric == "accuracy":
            results["accuracy"] = accuracy_score(y, y_pred)
        elif metric == "precision":
            results["precision"] = precision_score(y, y_pred, zero_division=0)
        elif metric == "recall":
            results["recall"] = recall_score(y, y_pred)
        else:
            logger.warning(f"⚠️ Unknown metric '{metric}' — skipping.")
    return results

# -------------------------------------------------------------------------
# Step 6 – extract top features for PSI monitoring
# -------------------------------------------------------------------------
def extract_feature_importance(model, feature_names, top_n=20):
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    mapped = [
        {"feature": feature_names[int(k.replace('f',''))], "gain": float(v)}
        for k, v in importance.items() if k.startswith('f')
    ]
    mapped = sorted(mapped, key=lambda x: x["gain"], reverse=True)
    top_feats = [x["feature"] for x in mapped[:top_n]]
    return mapped, top_feats

# -------------------------------------------------------------------------
# Step 7 – build and save model artefact
# -------------------------------------------------------------------------
def save_model(model, metadata, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.pkl")
    meta_path = os.path.join(save_dir, "metadata.json")
    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"💾 Saved model to {model_path}")
    logger.info(f"💾 Saved metadata to {meta_path}")

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main(snapshot_date=None, config_path="config/ML_config.yaml"):
    config = load_config(config_path)
    gold_dir = resolve_relative_path("datamart/gold")

    # load merged dataset
    max_date = datetime.strptime(snapshot_date, "%Y-%m-%d") 
    df, latest_snapshot = load_datasets(
        gold_dir,
        lag_months=4,
        min_months_required=9,
        max_snapshot_date=max_date # Prevents loading of future snapshots
    )
    if df is None:
        return

    snapshot_dates = sorted(df["snapshot_tag"].unique())
    split_cfg = config["splits"]
    train_df, val_df, test_df, oot_df = make_splits(df, snapshot_dates, split_cfg)


    target_col = "label"
    feature_cols = [c for c in df.columns if c not in [
        'Customer_ID', 'snapshot_date', 'loan_start_date', 'label', 'label_def', 'snapshot_tag'
    ]]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    X_oot, y_oot = oot_df[feature_cols], oot_df[target_col]

    # build preprocessor with metadata tracking
    preprocessor, auto_handled = build_preprocessor(config, X_train)
    preprocessor.fit(X_train)

    X_train_p = preprocessor.transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)
    X_oot_p = preprocessor.transform(X_oot)

    # --- Get feature names after preprocessing ---
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_ohe_names = list(ohe.get_feature_names_out(config["categorical_features"]))
    num_feats = list(config["imputation"]["numeric"].keys())
    
    # passthrough features (everything else not in imputation or categorical lists)
    passthrough_feats = [
        col for col in X_train.columns
        if col not in num_feats + config["categorical_features"]
    ]
    
    # combine all
    transformed_feature_names = num_feats + cat_ohe_names + passthrough_feats

    # train model
    best_model, best_params, cv_auc = train_xgb_model(X_train_p, y_train, config)

    # evaluate
    metrics_to_compute = config["evaluation"]["metrics"]
    results = {}
    for name, X_, y_ in zip(
        ["train", "val", "test", "oot"],
        [X_train_p, X_val_p, X_test_p, X_oot_p],
        [y_train, y_val, y_test, y_oot]
    ):
        results[name] = evaluate_model(best_model, X_, y_, metrics_to_compute)

    logger.info(f"📊 Metrics: {json.dumps(results, indent=2)}")

    # extract feature importance
    feat_importance, top_feats = extract_feature_importance(best_model, transformed_feature_names)

    # baseline feature stats for PSI
    baseline_stats = {}
    for feat in top_feats:
        if feat in X_train.columns:
            col_values = X_train[feat].dropna()
            baseline_stats[feat] = {"quantiles": list(col_values.quantile([0.1*i for i in range(1, 10)]).values)}

    # metadata
    model_artefact = {
        "model_version": f"credit_model_{latest_snapshot}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_window": [snapshot_dates[0], snapshot_dates[-1]],
        "hp_params": best_params,
        "cv_auc": cv_auc,
        "results": results,
        "monitoring": {
            "top_features_for_PSI": top_feats,
            "feature_importance_gain": feat_importance,
            "baseline_feature_stats": baseline_stats
        },
        "data_stats": {
            "rows_train": len(X_train),
            "rows_test": len(X_test),
            "rows_oot": len(X_oot),
            "y_rate_train": round(float(y_train.mean()), 3),
            "y_rate_test": round(float(y_test.mean()), 3),
            "y_rate_oot": round(float(y_oot.mean()), 3)
        }
    }

    # save under candidate_models/<snapshot_date>
    save_dir = resolve_relative_path(f"model_store/candidate_models/{latest_snapshot}")
    save_model(best_model, model_artefact, save_dir)

    # ---------------------------------------------------------------------
    # Save preprocessing artefacts for inference reuse
    # ---------------------------------------------------------------------
    preproc_dir = os.path.join(save_dir, "preprocessing")
    os.makedirs(preproc_dir, exist_ok=True)

    # Extract subcomponents
    num_pipe = preprocessor.named_transformers_["num"]
    cat_pipe = preprocessor.named_transformers_["cat"]
    num_imputer = num_pipe.named_steps.get("imputer")
    cat_imputer = cat_pipe.named_steps.get("imputer")
    encoder = cat_pipe.named_steps.get("encoder")

    # Save imputers & encoder
    joblib.dump(num_imputer, os.path.join(preproc_dir, "num_imputer.pkl"))
    joblib.dump(cat_imputer, os.path.join(preproc_dir, "cat_imputer.pkl"))
    joblib.dump(encoder, os.path.join(preproc_dir, "ohe_encoder.pkl"))

    # Save feature ordering information
    training_meta = {
        "numeric_features": list(num_imputer.feature_names_in_),
        "categorical_features": config["categorical_features"],
        "transformed_feature_names": transformed_feature_names,
    }
    with open(os.path.join(preproc_dir, "training_columns.json"), "w") as f:
        json.dump(training_meta, f, indent=2)

    logger.info(f"💾 Saved preprocessing artefacts to {preproc_dir}")

    logger.info(f"🎉 Model training completed successfully for snapshot={latest_snapshot}")

# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    snapshot_date = sys.argv[1] if len(sys.argv) > 1 else None
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/ML_config.yaml"
    main(snapshot_date, config_path)
