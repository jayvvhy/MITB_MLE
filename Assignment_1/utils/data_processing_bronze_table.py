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

def process_bronze_tables(raw_data_dir: str, bronze_dir: str, spark: SparkSession):
    """
    Process raw CSVs into Bronze Parquet tables, partitioned by snapshot_date.

    Args:
        raw_data_dir (str): Path to folder containing the raw CSVs.
        bronze_dir (str): Base path to save Bronze tables.
        spark (SparkSession): Active Spark session.
    """
    logger.info("Starting bronze layer processing...")
    
    # Map of raw filenames -> bronze table names
    datasets = {
        "features_attributes.csv": "features_attributes",
        "features_financials.csv": "features_financials",
        "feature_clickstream.csv": "feature_clickstream",
        "lms_loan_daily.csv": "lms_loan_daily"
    }

    for filename, table_name in datasets.items():
        csv_path = f"{raw_data_dir}/{filename}"

        print(f"Processing {filename}...")

        # Load CSV
        df = spark.read.csv(csv_path, header=True, inferSchema=True)

        # Ensure snapshot_date is DateType
        df = df.withColumn("snapshot_date", to_date(col("snapshot_date")))

        # Show row count per snapshot_date
        counts = df.groupBy("snapshot_date").count().orderBy("snapshot_date")
        print(f"📊 Row counts per snapshot_date for {table_name}:")
        for row in counts.collect():
            print(f"   {row['snapshot_date']}: {row['count']} rows")
        
        # Save to Bronze as partitioned Parquet (1 file per snapshot_date)
        output_path = f"{bronze_dir}/{table_name}"
        (
            df
            .coalesce(1)  # force 1 file per snapshot_date partition
            .write
            .mode("overwrite")
            .partitionBy("snapshot_date")
            .parquet(output_path)
        )

        print(f"✅ Saved Bronze table: {output_path}")
        logger.info("Bronze layer completed.")

