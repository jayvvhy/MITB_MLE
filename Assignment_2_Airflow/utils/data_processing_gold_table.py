import os
import sys
import yaml
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

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
# Helper: portable path resolver
# -------------------------------------------------------------------------
def resolve_path(path_str: str) -> str:
    """
    Resolve relative or absolute paths for both local and container runs.
    - Inside Docker → prefix with /opt/airflow
    - Locally → prefix with current working directory
    """
    if os.path.isabs(path_str):
        return path_str
    base_dir = "/opt/airflow" if os.path.exists("/opt/airflow") else os.getcwd()
    return os.path.join(base_dir, path_str.strip("/"))

# -------------------------------------------------------------------------
# Clickstream aggregation helper
# -------------------------------------------------------------------------
def aggregate_clickstream(clicks, feature_cols, loans):
    """Aggregate clickstream 3-month history (avg/sum) and last-month snapshot."""
    c, l = clicks.alias("c"), loans.select("Customer_ID", "loan_start_date").alias("l")
    joined = c.join(l, "Customer_ID")

    l3m = (
        joined
        .where(
            (F.col("c.snapshot_date") >= F.add_months(F.col("l.loan_start_date"), -3)) &
            (F.col("c.snapshot_date") < F.col("l.loan_start_date"))
        )
        .groupBy("c.Customer_ID", "l.loan_start_date")
        .agg(
            *[F.avg(F.col(f"c.{f}")).alias(f"avg_{f}_L3M") for f in feature_cols],
            *[F.sum(F.col(f"c.{f}")).alias(f"sum_{f}_L3M") for f in feature_cols]
        )
    )

    lm = (
        joined
        .where(F.col("c.snapshot_date") == F.add_months(F.col("l.loan_start_date"), -1))
        .groupBy("c.Customer_ID", "l.loan_start_date")
        .agg(*[F.first(F.col(f"c.{f}")).alias(f"{f}_LM") for f in feature_cols])
    )

    return l3m.join(lm, ["Customer_ID", "loan_start_date"], "left")

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def process_gold_tables(snapshot_date, spark, config_path="config/gold_config.yaml"):
    if not snapshot_date:
        raise ValueError("snapshot_date (YYYY-MM-DD) required.")
    snapshot_dt = datetime.strptime(snapshot_date, "%Y-%m-%d")

    # ---------------------------------------------------------------------
    # Resolve config path
    # ---------------------------------------------------------------------
    cfg_path = resolve_path(config_path)
    logger.info(f"🔍 Resolving config path: {cfg_path}")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------------------
    # Resolve paths from config
    # ---------------------------------------------------------------------
    silver_base = resolve_path(cfg["paths"]["silver_base"])
    gold_base = resolve_path(cfg["paths"]["gold_base"])
    static_cfg = cfg["static_features"]
    click_cols = cfg["clickstream_features"]
    loan_cfg = cfg["loan_types"]

    logger.info(f"🚀 Building Gold tables for {snapshot_date}")

    # ---------------------------------------------------------------------
    # Load current silver tables
    # ---------------------------------------------------------------------
    tag = snapshot_date.replace("-", "_")
    attrs = spark.read.parquet(os.path.join(silver_base, "features_attributes", f"{tag}.parquet"))
    fins = spark.read.parquet(os.path.join(silver_base, "features_financials", f"{tag}.parquet"))
    loans = spark.read.parquet(os.path.join(silver_base, "lms_loan_daily", f"{tag}.parquet"))

    # ---------------------------------------------------------------------
    # Load last 3 months of clickstream for L3M aggregation
    # ---------------------------------------------------------------------
    click_dfs = []
    missing_months = []
    for i in range(3, 0, -1):
        prev_tag = (snapshot_dt - relativedelta(months=i)).strftime("%Y_%m_%d")
        path = os.path.join(silver_base, "feature_clickstream", f"{prev_tag}.parquet")
        if os.path.exists(path):
            click_dfs.append(spark.read.parquet(path))
        else:
            missing_months.append(prev_tag)

    if missing_months:
        logger.warning(f"⚠️ Missing clickstream months: {missing_months}")

    if len(click_dfs) < 3:
        logger.info(f"⏭️ Skipping feature_store generation for {snapshot_date}: insufficient L3M clickstream data.")
        return None

    clicks = click_dfs[0]
    for df in click_dfs[1:]:
        clicks = clicks.unionByName(df, allowMissingColumns=True)

    # ---------------------------------------------------------------------
    # 1️⃣ Feature store (current snapshot loans)
    # ---------------------------------------------------------------------
    attr_cols = static_cfg["attributes"]
    fin_cols = static_cfg["financials"]

    static_feats = (
        attrs.alias("a")
        .join(fins.alias("f"), ["Customer_ID", "snapshot_date"], "inner")
        .select("Customer_ID", "snapshot_date", *attr_cols, *fin_cols)
    )

    feature_store = (
        loans.select("Customer_ID", "loan_start_date")
        .join(
            static_feats,
            (static_feats.Customer_ID == loans.Customer_ID) &
            (static_feats.snapshot_date == loans.loan_start_date),
            "left"
        )
        .drop(static_feats.Customer_ID, static_feats.snapshot_date)
    )

    if clicks is not None:
        click_aggs = aggregate_clickstream(clicks, click_cols, loans)
        feature_store = feature_store.join(click_aggs, ["Customer_ID", "loan_start_date"], "left")
        click_cols_expanded = [c for c in feature_store.columns if c.endswith("_L3M") or c.endswith("_LM")]
        feature_store = feature_store.fillna(0, subset=click_cols_expanded)

    # Multi-hot encode loan types
    expected_loans = loan_cfg["expected"]
    default_loan = loan_cfg["default"]
    feature_store = feature_store.withColumn(
        "Type_of_Loan_list",
        F.when(F.size(F.col("Type_of_Loan_list")) == 0, F.array(F.lit(default_loan)))
         .when(F.col("Type_of_Loan_list").isNull(), F.array(F.lit(default_loan)))
         .otherwise(F.col("Type_of_Loan_list"))
    )
    for loan in expected_loans + [default_loan]:
        feature_store = feature_store.withColumn(
            f"LoanType_{loan.replace(' ', '_')}",
            F.when(F.array_contains(F.col("Type_of_Loan_list"), loan), 1).otherwise(0)
        )
    feature_store = feature_store.drop("Type_of_Loan_list")

    # ---------------------------------------------------------------------
    # 2️⃣ Label store (loans whose 4 MOBs completed)
    # ---------------------------------------------------------------------
    mob_window = 4
    label_store = (
        loans.filter(F.col("mob") <= mob_window)
        .groupBy("Customer_ID", "loan_start_date")
        .agg(F.max(F.col("dpd_flag")).alias("label"))
        .withColumn("label_def", F.lit(f"30dpd_{mob_window}mob"))
        .filter(F.add_months(F.col("loan_start_date"), mob_window) <= F.lit(snapshot_dt))
        .withColumn("snapshot_date", F.lit(snapshot_dt))
    )

    # ---------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------
    feat_out = os.path.join(gold_base, "feature_store", f"{tag}.parquet")
    lab_out = os.path.join(gold_base, "label_store", f"{tag}.parquet")

    os.makedirs(os.path.dirname(feat_out), exist_ok=True)
    os.makedirs(os.path.dirname(lab_out), exist_ok=True)

    feature_store.coalesce(1).write.mode("overwrite").parquet(feat_out)
    label_store.coalesce(1).write.mode("overwrite").parquet(lab_out)

    logger.info(f"✅ Feature store saved: {feat_out}")
    logger.info(f"✅ Label store saved:   {lab_out}")
    logger.info(f"🎉 Completed Gold ETL for {snapshot_date}")
    return True

# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process Gold Tables")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format")
    parser.add_argument("--config_path", default="config/gold_config.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("GoldLayerProcessing").getOrCreate()

    process_gold_tables(
        snapshot_date=args.snapshot_date,
        spark=spark,
        config_path=args.config_path
    )

    spark.stop()
