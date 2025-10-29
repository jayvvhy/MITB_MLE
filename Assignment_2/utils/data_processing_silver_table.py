import os
import sys
import glob
import shutil
import yaml
import logging
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, when, lit, regexp_replace, split, size, coalesce,
    array, regexp_extract, round, to_date
)
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def resolve_relative_path(path: str) -> str:
    """
    Converts a relative path (from YAML) to an absolute path
    relative to the project root (one level up from /utils).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # assumes structure: /app/utils, /app/datamart, /app/config
    abs_path = os.path.abspath(os.path.join(base_dir, path))
    return abs_path

# -------------------------------------------------------------------------
# Helper cleaning functions
# -------------------------------------------------------------------------
def clean_numeric_column(df: DataFrame, col_name: str, min_val=None, max_val=None, remove_trailing_underscore=True) -> DataFrame:
    temp_col = f"{col_name}_cleaned"
    if remove_trailing_underscore:
        df = df.withColumn(temp_col, regexp_replace(col(col_name).cast("string"), "_", ""))
    else:
        df = df.withColumn(temp_col, col(col_name).cast("string"))
    df = df.withColumn(temp_col, when(col(temp_col) == "", None).otherwise(col(temp_col).cast(FloatType())))
    if min_val is not None:
        df = df.withColumn(temp_col, when(col(temp_col) < min_val, None).otherwise(col(temp_col)))
    if max_val is not None:
        df = df.withColumn(temp_col, when(col(temp_col) > max_val, lit(max_val)).otherwise(col(temp_col)))
    df = df.withColumn(col_name, col(temp_col)).drop(temp_col)
    return df

def clean_string_column(df: DataFrame, col_name: str, junk_values: list = None, placeholder=None) -> DataFrame:
    temp_col = col_name
    if junk_values:
        for junk in junk_values:
            df = df.withColumn(temp_col, when(col(temp_col) == junk, None).otherwise(col(temp_col)))
    if placeholder:
        df = df.withColumn(temp_col, when(col(temp_col) == placeholder, None).otherwise(col(temp_col)))
    return df

def clean_features_attributes(df: DataFrame) -> DataFrame:
    df = clean_numeric_column(df, "Age", min_val=0, max_val=120)
    df = df.withColumn("Age_valid", when((col("Age") >= 0) & (col("Age") <= 120), lit(1)).otherwise(lit(0)))
    df = df.withColumn("SSN", when(col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"), col("SSN")).otherwise(lit(None)))
    df = clean_string_column(df, "Occupation", placeholder="_______")
    return df

def clean_features_financials(df: DataFrame) -> DataFrame:
    df = clean_numeric_column(df, "Annual_Income")
    df = df.withColumn("Annual_Income", round(col("Annual_Income"), 2))
    df = clean_numeric_column(df, "Num_Bank_Accounts", min_val=0, max_val=11)
    df = clean_numeric_column(df, "Num_Credit_Card", max_val=11)
    df = clean_numeric_column(df, "Interest_Rate", max_val=34)
    df = clean_numeric_column(df, "Num_of_Delayed_Payment", max_val=28)
    df = df.withColumn("Changed_Credit_Limit", when(col("Changed_Credit_Limit") == "_", None)
                       .otherwise(round(col("Changed_Credit_Limit").cast("float"), 2)))
    df = clean_string_column(df, "Credit_Mix", junk_values=["_"])
    df = clean_numeric_column(df, "Outstanding_Debt")
    df = df.withColumn("Credit_History_Years", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast(IntegerType()))
    df = df.withColumn("Credit_History_Months", regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast(IntegerType()))
    df = df.withColumn("Credit_History_Age_Months", (col("Credit_History_Years")*12 + col("Credit_History_Months")).cast(IntegerType()))
    df = df.withColumn(
        "Type_of_Loan_list",
        when(
            col("Type_of_Loan").isNotNull(),
            split(regexp_replace(col("Type_of_Loan"), r"\band\s+", ""), r",\s*")
        ).otherwise(array())
    )
    df = df.withColumn("Num_of_Loan", coalesce(size("Type_of_Loan_list"), lit(0)))
    df = df.withColumn("Amount_invested_monthly",
                       regexp_replace(col("Amount_invested_monthly").cast("string"), "^_+|_+$", "").cast(FloatType()))
    df = clean_string_column(df, "Payment_Behaviour", junk_values=["!@9#%8"])
    df = clean_string_column(df, "Monthly_Balance", junk_values=["__-333333333333333333333333333__"])
    return df

def augment_lms_loan_daily(df: DataFrame) -> DataFrame:
    df = df.withColumn("mob", F.col("installment_num"))
    df = df.withColumn("dpd_flag", F.when(F.col("overdue_amt") > 0, 1).otherwise(0))
    df = df.withColumn("loan_start_month", F.trunc(F.col("loan_start_date"), "month"))
    df = df.withColumn("snapshot_month", F.trunc(F.col("snapshot_date"), "month"))
    return df

def no_cleaning(df: DataFrame) -> DataFrame:
    return df

# -------------------------------------------------------------------------
# Main processing function
# -------------------------------------------------------------------------
def process_silver_tables(snapshot_date: str, spark: SparkSession, config_path: str = "config/silver_config.yaml"):
    """
    Process Bronze Parquet tables into Silver layer for a specific snapshot_date.
    Reads config YAML (resolved relative to this script).
    """
    if snapshot_date is None:
        raise ValueError("snapshot_date must be provided (format: YYYY-MM-DD).")

    # --- Resolve config path relative to this script ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # assume structure /app/utils + /app/config
    resolved_path = os.path.join(base_dir, config_path)
    resolved_path = os.path.abspath(resolved_path)
    logger.info(f"🔍 Resolving config path: {resolved_path}")

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path, "r") as f:
        config = yaml.safe_load(f)

    bronze_dir = resolve_relative_path(config["directories"]["bronze_dir"])
    silver_dir = resolve_relative_path(config["directories"]["silver_dir"])

    logger.info(f"🚀 Starting silver layer processing for snapshot_date={snapshot_date}")
    logger.info(f"Using configuration from: {resolved_path}")

    snapshot_date_str = snapshot_date.replace("-", "_")

    # ---------------------------------------------------------------------
    # Process each dataset defined in config
    # ---------------------------------------------------------------------
    for table_name, dataset_config in config["datasets"].items():
        logger.info(f"📂 Processing Silver table: {table_name}...")

        bronze_path = os.path.join(bronze_dir, table_name, f"{snapshot_date_str}.parquet")
        if not os.path.exists(bronze_path):
            logger.warning(f"⚠️ No Bronze file found for {table_name} on {snapshot_date}. Skipping...")
            continue

        df = spark.read.parquet(bronze_path)

        # Drop rows missing keys
        for key in dataset_config.get("drop_nulls", []):
            df = df.filter(col(key).isNotNull())

        # Apply dataset-specific cleaning
        if table_name == "features_attributes":
            df = clean_features_attributes(df)
        elif table_name == "features_financials":
            df = clean_features_financials(df)
        elif table_name == "lms_loan_daily":
            df = augment_lms_loan_daily(df)
        else:
            df = no_cleaning(df)            

        # Cast columns to defined types
        type_map = {"string": StringType(), "int": IntegerType(), "float": FloatType(), "date": DateType()}
        for col_name, dtype_str in dataset_config.get("types", {}).items():
            if col_name in df.columns:
                df = df.withColumn(col_name, col(col_name).cast(type_map[dtype_str]))

        # --- Write to Silver ---
        table_output_dir = os.path.join(silver_dir, table_name)
        os.makedirs(table_output_dir, exist_ok=True)

        temp_output_dir = os.path.join(table_output_dir, f"tmp_{snapshot_date_str}")
        final_output_path = os.path.join(table_output_dir, f"{snapshot_date_str}.parquet")

        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir, ignore_errors=True)

        (
            df.coalesce(1)
              .write
              .mode("overwrite")
              .parquet(temp_output_dir)
        )

        parquet_files = glob.glob(os.path.join(temp_output_dir, "part-*.parquet"))
        if parquet_files:
            shutil.move(parquet_files[0], final_output_path)
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            logger.info(f"✅ Saved Silver file: {final_output_path}")
        else:
            logger.warning(f"⚠️ No parquet file written for {table_name} ({snapshot_date}).")

    logger.info("🎉 Silver layer processing completed successfully.")

# -------------------------------------------------------------------------
# Script entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    spark = SparkSession.builder.appName("SilverLayerProcessing").getOrCreate()

    snapshot_date = sys.argv[1] if len(sys.argv) > 1 else None
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/silver_config.yaml"

    process_silver_tables(
        snapshot_date=snapshot_date,
        spark=spark,
        config_path=config_path
    )

    spark.stop()
