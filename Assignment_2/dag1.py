from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

bronze_task = SparkSubmitOperator(
    task_id="process_bronze_snapshot",
    application="/opt/airflow/dags/data_processing_bronze_table.py",
    application_args=["{{ ds }}", "/config/bronze_config.yaml"],
)
