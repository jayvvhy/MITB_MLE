import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, to_date, when, lit, coalesce, size, split, explode, collect_set, array_contains, regexp_replace, round, regexp_extract
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
import re
import yaml

import logging

logger = logging.getLogger(__name__)

def aggregate_clickstream(clicks, feature_cols, loans):
    """
    Aggregates clickstream data (L3M avg/sum and LM values).
    """

    c = clicks.alias("c")
    l = loans.select("Customer_ID", "loan_start_date").alias("l")

    joined = c.join(l, on="Customer_ID")

    # ---- L3M: avg & sum over last 3 months strictly before loan_start_date ----
    l3m = (
        joined
        .where((F.col("c.snapshot_date") >= F.add_months(F.col("l.loan_start_date"), -3)) &
               (F.col("c.snapshot_date") <  F.col("l.loan_start_date")))
        .groupBy(
            F.col("c.Customer_ID").alias("Customer_ID"),
            F.col("l.loan_start_date").alias("loan_start_date")
        )
        .agg(
            *[F.avg(F.col(f"c.{col}")).alias(f"avg_{col}_L3M") for col in feature_cols],
            *[F.sum(F.col(f"c.{col}")).alias(f"sum_{col}_L3M") for col in feature_cols]
        )
    )

    # ---- LM: exact value for snapshot_date == loan_start_date - 1 month
    # Use groupBy + first() to guard against multiple rows per month (just take first)
    lm = (
        joined
        .where(F.col("c.snapshot_date") == F.add_months(F.col("l.loan_start_date"), -1))
        .groupBy(
            F.col("c.Customer_ID").alias("Customer_ID"),
            F.col("l.loan_start_date").alias("loan_start_date")
        )
        .agg(
            *[F.first(F.col(f"c.{col}")).alias(f"{col}_LM") for col in feature_cols]
        )
    )

    # Join L3M and LM (one row per key)
    click_aggs = l3m.join(lm, on=["Customer_ID", "loan_start_date"], how="left")

    return click_aggs

def process_gold_tables(spark, config_path="/app/config/gold_config.yaml"):
    """
    Build gold tables using YAML-driven configuration.
    Important note: Identifiers "Customer_ID" and "loan_start_date" are retained in
    feature and label stores to faciitate mapping of output to prediction. They will
    be dropped as part of the ML pipeline during training.
    """
    logger.info("Starting gold layer processing...")
    # ----------------------------------------------------------------------
    # 1. Load YAML configuration
    # ----------------------------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    silver_base_path = config["paths"]["silver_base"]
    gold_base_path = config["paths"]["gold_base"]
    splits_cfg = config["splits"]
    static_cfg = config["static_features"]
    click_cols = config["clickstream_features"]
    loan_cfg = config["loan_types"]
    cat_cols = config["categorical_features"]
    impute_cfg = config["imputation"]
    
    # ----------------------------------------------------------------------
    # 2. Load silver datasets
    # ----------------------------------------------------------------------
    attrs = spark.read.parquet(f"{silver_base_path}/features_attributes")
    fins = spark.read.parquet(f"{silver_base_path}/features_financials")
    clicks = spark.read.parquet(f"{silver_base_path}/feature_clickstream")
    loans = spark.read.parquet(f"{silver_base_path}/lms_loan_daily")

    # ----------------------------------------------------------------------
    # 3. Filter away early loans (truncate on or before 2023-03-01)
    # ----------------------------------------------------------------------
    loans = loans.filter(F.col("loan_start_date") > F.to_date(F.lit(splits_cfg["min_date"])))

    # ----------------------------------------------------------------------
    # 4. Label store: binary default flag
    # ----------------------------------------------------------------------
    label_store = (
        loans.groupBy("Customer_ID", "loan_start_date")
             .agg(F.max(F.when(F.col("overdue_amt") > 0, 1).otherwise(0)).alias("default_flag"))
    )

    # ----------------------------------------------------------------------
    # 5. Static features (attributes + financials)
    # ----------------------------------------------------------------------
    attr_cols = static_cfg["attributes"]
    fin_cols = static_cfg["financials"]

    static_feats = (
        attrs.alias("a")
             .join(fins.alias("f"), on=["Customer_ID", "snapshot_date"], how="inner")
             .select("Customer_ID", "snapshot_date", *attr_cols, *fin_cols)
    )

    # ----------------------------------------------------------------------
    # 6. Clickstream aggregation (LM / L3M)
    # ----------------------------------------------------------------------

    click_aggs = aggregate_clickstream(clicks, click_cols, loans)

    # ----------------------------------------------------------------------
    # 7. Feature store = static + clickstream, aligned with label store loans
    # ----------------------------------------------------------------------
    feature_store = (
        label_store.select("Customer_ID", "loan_start_date")  # anchor keys
        .join(static_feats,
              (static_feats.Customer_ID == label_store.Customer_ID) &
              (static_feats.snapshot_date == label_store.loan_start_date),
              "left")
        .drop(static_feats.Customer_ID)
        .drop(static_feats.snapshot_date)
        .join(click_aggs, on=["Customer_ID", "loan_start_date"], how="left")
    )
    
    # --- Handle missing clickstream features ---
    click_cols = [c for c in feature_store.columns if c.endswith("_L3M") or c.endswith("_LM")]
    feature_store = feature_store.fillna(0, subset=click_cols)    

    # ----------------------------------------------------------------------
    # 8. Define date-based splits
    # ----------------------------------------------------------------------
    max_date = feature_store.agg(F.max("loan_start_date")).first()[0]
    oot_months = splits_cfg["oot_months"]
    test_months = splits_cfg["test_months"]
    val_months = splits_cfg["val_months"]
    train_months = splits_cfg["train_months"]

    oot_start = max_date
    test_end = F.add_months(F.lit(oot_start), -oot_months)
    test_start = F.add_months(test_end, -test_months + 1)
    val_end = F.add_months(test_start, -1)
    val_start = F.add_months(val_end, -val_months + 1)
    train_start = F.add_months(val_start, -train_months + 1)

    train_df = feature_store.filter((F.col("loan_start_date") >= train_start) & (F.col("loan_start_date") <= val_start))
    val_df = feature_store.filter((F.col("loan_start_date") >= val_start) & (F.col("loan_start_date") <= val_end))
    test_df = feature_store.filter((F.col("loan_start_date") >= test_start) & (F.col("loan_start_date") <= test_end))
    oot_df = feature_store.filter(F.col("loan_start_date") == oot_start)

    # ----------------------------------------------------------------------
    # 9. Label alignment
    # ----------------------------------------------------------------------
    def align_labels(df):
        return label_store.join(df.select("Customer_ID", "loan_start_date"), on=["Customer_ID", "loan_start_date"])

    train_labels = align_labels(train_df)
    val_labels = align_labels(val_df)
    test_labels = align_labels(test_df)
    oot_labels = align_labels(oot_df)
    
    # ----------------------------------------------------------------------
    # --- ML specific handling ---
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # 10. Missing value handling
    # ----------------------------------------------------------------------
    num_impute = impute_cfg["numeric"]
    cat_impute = impute_cfg["categorical"]

    # Compute median values from training data
    median_vals = {}
    for col, method in num_impute.items():
        if method == "median":
            val = train_df.approxQuantile(col, [0.5], 0.01)
            median_vals[col] = val[0] if val else None
        else:
            median_vals[col] = method  # numeric constant

    def impute_missing(df):
        for col, val in median_vals.items():
            df = df.withColumn(col, F.when(F.col(col).isNull(), F.lit(val)).otherwise(F.col(col)))
        for col, val in cat_impute.items():
            df = df.withColumn(col, F.when(F.col(col).isNull(), F.lit(val)).otherwise(F.col(col)))
        return df

    train_df = impute_missing(train_df)
    val_df = impute_missing(val_df)
    test_df = impute_missing(test_df)
    oot_df = impute_missing(oot_df)

    # ----------------------------------------------------------------------
    # 11. Multi-hot encoding for Type_of_Loan_list
    # ----------------------------------------------------------------------
    expected_loans = loan_cfg["expected"]
    default_loan = loan_cfg["default"]

    def encode_loans(df):
        df = df.withColumn(
            "Type_of_Loan_list",
            F.when(F.size(F.col("Type_of_Loan_list")) == 0, F.array(F.lit(default_loan)))
             .when(F.col("Type_of_Loan_list").isNull(), F.array(F.lit(default_loan)))
             .otherwise(F.col("Type_of_Loan_list"))
        )
        unique_vals = [row[0] for row in df.select(F.explode("Type_of_Loan_list")).distinct().collect()]
        unexpected = set(unique_vals) - set(expected_loans) - {default_loan}
        if unexpected:
            print(f"⚠️ Unexpected loan types found: {unexpected}")
        else:
            print(f"✅ No unexpected loan types found in data.")

        for loan in expected_loans + [default_loan]:
            df = df.withColumn(f"LoanType_{loan.replace(' ', '_')}",
                               F.when(F.array_contains(F.col("Type_of_Loan_list"), loan), 1).otherwise(0))
        return df.drop("Type_of_Loan_list")

    train_df = encode_loans(train_df)
    val_df = encode_loans(val_df)
    test_df = encode_loans(test_df)
    oot_df = encode_loans(oot_df)

    # ----------------------------------------------------------------------
    # 12. One-hot encoding for other categorical variables (fit on train only)
    # ----------------------------------------------------------------------
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_OHE") for c in cat_cols]

    pipeline = Pipeline(stages=indexers + encoders)
    fitted_pipeline = pipeline.fit(train_df)

    def apply_ohe(df):
        df = fitted_pipeline.transform(df)
        for c in cat_cols:
            labels = fitted_pipeline.stages[cat_cols.index(c)].labels
            arr = vector_to_array(f"{c}_OHE")
            df = df.withColumn(f"{c}_arr", arr)
            for i, label in enumerate(labels):
                clean_label = label.replace(" ", "_").replace("/", "_").replace("-", "_")
                df = df.withColumn(f"{c}_{clean_label}", F.col(f"{c}_arr")[i])
            df = df.drop(f"{c}_arr", f"{c}_OHE", f"{c}_idx", c)
        return df

    train_df = apply_ohe(train_df)
    val_df = apply_ohe(val_df)
    test_df = apply_ohe(test_df)
    oot_df = apply_ohe(oot_df)

    # ----------------------------------------------------------------------
    # 13. Drop identifiers --- Identifiers will only be dropped during training
    # ----------------------------------------------------------------------
    # for df in [train_df, val_df, test_df, oot_df]:
    #     df.drop("Customer_ID", "loan_start_date")

    # train_labels = train_labels.drop("Customer_ID", "loan_start_date")
    # val_labels = val_labels.drop("Customer_ID", "loan_start_date")
    # test_labels = test_labels.drop("Customer_ID", "loan_start_date")
    # oot_labels = oot_labels.drop("Customer_ID", "loan_start_date")

    # ----------------------------------------------------------------------
    # 14. Save splits
    # ----------------------------------------------------------------------
    splits = {
        "train": (train_df, train_labels),
        "val": (val_df, val_labels),
        "test": (test_df, test_labels),
        "OOT": (oot_df, oot_labels),
    }

    for name, (feat, lab) in splits.items():
        print(f"📊 {name.upper()} split: features shape=({feat.count()}, {len(feat.columns)}), labels shape=({lab.count()}, {len(lab.columns)})")
        feat.write.mode("overwrite").parquet(f"{gold_base_path}/feature_store/{name}")
        lab.write.mode("overwrite").parquet(f"{gold_base_path}/label_store/{name}")
        print(f"✅ {name} split saved.")
    logger.info("Gold layer completed.")

    return splits