import requests
import dask.dataframe as dd
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, count
from pyspark.sql.window import Window
import pickle

# Initialize Spark session
spark = SparkSession.builder.appName("EmployeeDataAnalysis").getOrCreate()

# API URL
api_url = "http://127.0.0.1:5000/fetch_all_data"

# Step 1: Fetch data from the API and preprocess with Dask
def preprocess_with_dask(api_url):
    # Fetch data from the API
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    # Load data into a Dask DataFrame
    data = response.json()
    dask_df = dd.from_pandas(pd.DataFrame(data), npartitions=4)

    # Ensure proper data types
    dask_df['age'] = dask_df['age'].astype('int32')
    dask_df['performance_score'] = dask_df['performance_score'].astype('float32')
    dask_df['job_satisfaction'] = dask_df['job_satisfaction'].astype('float32')
    dask_df['work_hours'] = dask_df['work_hours'].astype('int32')

    # Convert 'timestamp' column using Dask's to_datetime method
    dask_df['timestamp'] = dd.to_datetime(dask_df['timestamp'], format="%Y-%m-%d %H:%M:%S")

    # Feature Engineering
    dask_df['productivity_index'] = (
        dask_df['performance_score'] * dask_df['job_satisfaction'] / dask_df['work_hours']
    )
    dask_df['attrition_risk_numeric'] = dask_df['attrition_risk'].map({
        "Low": 1, "Medium": 2, "High": 3
    })

    return dask_df

# Step 2: Convert Dask DataFrame to PySpark DataFrame
def load_to_spark(dask_df, spark):
    # Convert Dask DataFrame to Pandas, then to PySpark
    pandas_df = dask_df.compute()  # Trigger computation in Dask and convert to Pandas
    spark_df = spark.createDataFrame(pandas_df)
    return spark_df

# Step 3: Advanced PySpark Analysis
def analyze_with_spark(spark_df):
    # Advanced Aggregations
    print("=== Department-Wise Performance Metrics ===")
    department_metrics_df = spark_df.groupBy("department").agg(
        avg("performance_score").alias("avg_performance_score"),
        avg("job_satisfaction").alias("avg_job_satisfaction"),
        sum("work_hours").alias("total_work_hours")
    )
    department_metrics_df.show()

    # Save department metrics to pickle
    department_metrics_pd = department_metrics_df.toPandas()
    with open("department_metrics.pkl", "wb") as f:
        pickle.dump(department_metrics_pd, f)

    # Anomaly Detection (e.g., high attrition risk)
    print("=== Anomalies (High Attrition Risk) ===")
    anomalies_df = spark_df.filter(col("attrition_risk_numeric") == 3)
    anomalies_df.show()

    # Save anomalies to pickle
    anomalies_df_pd = anomalies_df.toPandas()
    with open("anomalies.pkl", "wb") as f:
        pickle.dump(anomalies_df_pd, f)

    # Rolling Averages for Productivity Index
    print("=== Rolling Averages (Productivity Index) ===")
    window_spec = Window.orderBy("timestamp").rowsBetween(-5, 5)  # 10-record rolling window
    rolling_avg_df = spark_df.withColumn("rolling_avg_productivity", avg("productivity_index").over(window_spec))
    rolling_avg_df.show()

    # Save rolling averages to pickle
    rolling_avg_df_pd = rolling_avg_df.toPandas()
    with open("rolling_avg.pkl", "wb") as f:
        pickle.dump(rolling_avg_df_pd, f)

    # Correlation Analysis
    print("=== Correlation Between Productivity and Attrition Risk ===")
    correlation = spark_df.stat.corr("productivity_index", "attrition_risk_numeric")
    print(f"Correlation between Productivity Index and Attrition Risk: {correlation}")

    # Save correlation result to pickle
    correlation_result = {"correlation": correlation}
    with open("correlation.pkl", "wb") as f:
        pickle.dump(correlation_result, f)

    # Event-Based Distribution
    print("=== Distribution of Employee Roles ===")
    role_distribution = spark_df.groupBy("role").count()
    role_distribution.show()

    # Save role distribution to pickle
    role_distribution_pd = role_distribution.toPandas()
    with open("role_distribution.pkl", "wb") as f:
        pickle.dump(role_distribution_pd, f)

# Step 4: Main flow
if __name__ == "__main__":
    # Step 4.1: Preprocess with Dask
    dask_processed_df = preprocess_with_dask(api_url)

    # Step 4.2: Load data into PySpark
    spark_df = load_to_spark(dask_processed_df, spark)

    # Step 4.3: Analyze data with PySpark
    analyze_with_spark(spark_df)
