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
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_date, when, lit, coalesce, size, split, explode, collect_set, array_contains, array, regexp_replace, round, regexp_extract
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
import re
import yaml

import logging

logger = logging.getLogger(__name__)

import yaml
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, lit, regexp_replace, split, size, coalesce, array, regexp_extract, round
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# === Helper functions ===

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

# Dataset-specific cleaning functions (original working ones)
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

def no_cleaning(df: DataFrame) -> DataFrame:
    return df

# === Main processing function ===

def process_silver_tables(spark: SparkSession, config_path="/app/config/silver_config.yaml"):
    logger.info("Starting silver layer processing...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    bronze_dir = config["directories"]["bronze_dir"]
    silver_dir = config["directories"]["silver_dir"]

    for table_name, dataset_config in config["datasets"].items():
        print(f"Processing Silver table: {table_name}...")

        # Load Bronze
        df = spark.read.parquet(f"{bronze_dir}/{dataset_config['path']}")

        # Drop rows missing keys
        for key in dataset_config.get("drop_nulls", []):
            df = df.filter(col(key).isNotNull())

        # Apply cleaning logic
        if table_name == "features_attributes":
            df = clean_features_attributes(df)
        elif table_name == "features_financials":
            df = clean_features_financials(df)
        elif table_name in ["feature_clickstream", "lms_loan_daily"]:
            df = no_cleaning(df)

        # Cast types
        type_map = {"string": StringType(), "int": IntegerType(), "float": FloatType(), "date": DateType()}
        for col_name, dtype_str in dataset_config.get("types", {}).items():
            if col_name in df.columns:
                df = df.withColumn(col_name, col(col_name).cast(type_map[dtype_str]))

        # Write to Silver
        output_path = f"{silver_dir}/{table_name}"
        df.write.mode("overwrite").partitionBy("snapshot_date").parquet(output_path)
        print(f"✅ Saved Silver table: {output_path}")
        logger.info("Silver layer completed.")
