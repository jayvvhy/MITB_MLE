import os
import sys
import glob
import shutil
import yaml
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# -------------------------------------------------------------------------
# Main processing function
# -------------------------------------------------------------------------
def process_bronze_tables(snapshot_date: str, spark: SparkSession, config_path: str = "config/bronze_config.yaml"):
    """
    Process raw CSVs into Bronze Parquet tables for a specific snapshot_date.
    Configuration (paths and dataset list) are loaded from a YAML file.

    Args:
        snapshot_date (str): Snapshot date (YYYY-MM-DD) to filter on.
        spark (SparkSession): Active Spark session.
        config_path (str): Path to YAML configuration file (default: config/bronze_config.yaml).
    """
    if snapshot_date is None:
        raise ValueError("snapshot_date must be provided (format: YYYY-MM-DD).")

    # ---------------------------------------------------------------------
    # Resolve config path relative to this script, not working directory
    # ---------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # move one folder up to /app
    resolved_path = os.path.join(base_dir, config_path)
    resolved_path = os.path.abspath(resolved_path)

    logger.info(f"🔍 Resolving config path: {resolved_path}")

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path, "r") as f:
        config = yaml.safe_load(f)

    raw_data_dir = config["raw_data_dir"]
    bronze_dir = config["bronze_dir"]
    datasets = config["datasets"]

    logger.info(f"🚀 Starting bronze processing for snapshot_date={snapshot_date}")
    logger.info(f"Using configuration from: {resolved_path}")
    snapshot_date_str = snapshot_date.replace("-", "_")

    # ---------------------------------------------------------------------
    # Process each dataset listed in the config
    # ---------------------------------------------------------------------
    for filename, table_name in datasets.items():
        csv_path = os.path.join(raw_data_dir, filename)
        logger.info(f"📂 Processing file: {csv_path}")

        if not os.path.exists(csv_path):
            logger.warning(f"⚠️ File not found: {csv_path}. Skipping...")
            continue

        # Read CSV
        df = spark.read.csv(csv_path, header=True, inferSchema=True)

        # Ensure snapshot_date column exists
        if "snapshot_date" not in df.columns:
            raise ValueError(f"'snapshot_date' column missing in {filename}")

        # Filter for the given snapshot_date
        df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))
        df_filtered = df.filter(col("snapshot_date") == snapshot_date)

        count_filtered = df_filtered.count()
        logger.info(f"📅 {table_name}: {count_filtered} records found for {snapshot_date}")

        if count_filtered == 0:
            logger.warning(f"No records found for {table_name} on {snapshot_date}. Skipping...")
            continue

        # -----------------------------------------------------------------
        # Define output paths and write to Parquet
        # -----------------------------------------------------------------
        table_output_dir = os.path.join(bronze_dir, table_name)
        os.makedirs(table_output_dir, exist_ok=True)

        temp_output_dir = os.path.join(table_output_dir, f"tmp_{snapshot_date_str}")
        final_output_path = os.path.join(table_output_dir, f"{snapshot_date_str}.parquet")

        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir, ignore_errors=True)

        (
            df_filtered
            .coalesce(1)
            .write
            .mode("overwrite")
            .parquet(temp_output_dir)
        )

        # Rename the single parquet file from Spark’s temp directory
        parquet_files = glob.glob(os.path.join(temp_output_dir, "part-*.parquet"))
        if not parquet_files:
            logger.error(f"No parquet file generated for {table_name} ({snapshot_date}).")
            continue

        shutil.move(parquet_files[0], final_output_path)
        shutil.rmtree(temp_output_dir, ignore_errors=True)

        logger.info(f"✅ Saved Bronze file: {final_output_path} ({count_filtered} rows)")

    logger.info("🎉 Bronze layer processing completed successfully.")

# -------------------------------------------------------------------------
# Script entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    spark = SparkSession.builder.appName("BronzeLayerProcessing").getOrCreate()

    snapshot_date = sys.argv[1] if len(sys.argv) > 1 else None
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/bronze_config.yaml"

    process_bronze_tables(
        snapshot_date=snapshot_date,
        spark=spark,
        config_path=config_path
    )

    spark.stop()
