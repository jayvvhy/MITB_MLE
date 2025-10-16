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
import sys

from utils.data_processing_bronze_table import process_bronze_tables
from utils.data_processing_silver_table import process_silver_tables
from utils.data_processing_gold_table import process_gold_tables

# --------------------------------------------------------------------
# 🧱 LOGGING SETUP
# --------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"/app/logs/preprocessing_{timestamp}.log"

# Configure root logger
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Add console handler (so logs show in notebook/terminal too)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)
logger.info("Pipeline run started.")

# --------------------------------------------------------------------
# 🪶 Redirect print() and errors to logger
# --------------------------------------------------------------------
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip():
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

# --------------------------------------------------------------------
# ⚙️ SPARK INITIALIZATION
# --------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("dev")
    .master("local[*]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# --------------------------------------------------------------------
# 📦 PATH CONFIGURATION
# --------------------------------------------------------------------
raw_data_dir = "/app/data"
bronze_dir = "/app/datamart/bronze"

# --------------------------------------------------------------------
# 🚀 PIPELINE EXECUTION
# --------------------------------------------------------------------
try:
    process_bronze_tables(raw_data_dir, bronze_dir, spark)
    process_silver_tables(spark)
    splits = process_gold_tables(spark)

except Exception as e:
    logger.exception("Pipeline run failed due to an error.")
else:
    logger.info("Pipeline run completed successfully.")