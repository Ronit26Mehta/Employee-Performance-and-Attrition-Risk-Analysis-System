import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.ensemble import IsolationForest
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Initialize Spark Session
spark = SparkSession.builder.appName("EmployeeFinancialML").getOrCreate()

# API URL
api_url = "http://127.0.0.1:5000/recent_data"  # Replace with your API endpoint

# Function to map categorical data to numeric codes
def map_categorical_data(df):
    """
    This function maps categorical columns to numeric codes using the category codes.
    It ensures all categorical columns are converted to numeric values.
    """
    # Identify categorical columns (string/object types)
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Apply category code mapping for each categorical column
    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes
    
    return df

# Function to preprocess data (convert to numeric, fill missing values)
def preprocess_data(df):
    """
    This function converts all columns to numeric types and handles missing values.
    It applies category code mapping to categorical columns and converts all columns to numeric.
    Missing values are replaced by the column mean.
    """
    # First map categorical columns to numeric
    df = map_categorical_data(df)
    
    # Convert all columns to numeric where possible (non-categorical)
    for col in df.columns:
        # If the column is not already numeric, attempt conversion to numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, replace errors with NaN
    
    # Fill missing values with the mean of the column
    df.fillna(df.mean(), inplace=True)
    
    return df

# Fetch recent data from the API
def fetch_recent_data(api_url, num_records=100):
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}")
    data = response.json()
    df = pd.DataFrame(data[-num_records:])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)  # Set timestamp as the index
    return df

# Convert Pandas DataFrame to Spark DataFrame
def convert_to_spark_df(df):
    spark_df = spark.createDataFrame(df)
    assembler = VectorAssembler(
        inputCols=["performance_score", "job_satisfaction", "attrition_risk", "department", "role", "promotion_status"],
        outputCol="features"
    )
    spark_df = assembler.transform(spark_df)
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    spark_df = scaler.fit(spark_df).transform(spark_df)
    return spark_df

# Train a classification model and save predictions
def train_classifier(spark_df):
    spark_df = spark_df.withColumn("label", spark_df["attrition_risk"])
    train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=1234)
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label")
    model = rf.fit(train_data)
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    
    # Save predictions to pickle file
    predictions_df = predictions.toPandas()
    predictions_df.to_pickle('predictions.pkl')
    return predictions_df, accuracy

# Detect anomalies using Isolation Forest and save anomalies
def detect_anomalies(df):
    features = df[["performance_score", "job_satisfaction", "attrition_risk"]].values
    model = IsolationForest(contamination=0.1)
    anomalies = model.fit_predict(features)
    df['anomaly'] = anomalies
    
    # Save anomalies to pickle file
    anomalies_df = df[df['anomaly'] == -1]
    anomalies_df.to_pickle('anomalies.pkl')
    return anomalies_df

# Train a clustering model and save results
def train_clustering(spark_df):
    kmeans = KMeans(k=3, featuresCol="scaled_features", predictionCol="cluster")
    model = kmeans.fit(spark_df)
    predictions = model.transform(spark_df)
    
    # Save clustering results to pickle file
    clustering_df = predictions.toPandas()
    clustering_df.to_pickle('clusters.pkl')
    return clustering_df

# Main function to run the ML tasks and save models to pickle files
def perform_ml_operations():
    # Fetch and preprocess data
    df = fetch_recent_data(api_url)
    df = preprocess_data(df)
    spark_df = convert_to_spark_df(df)
    
    # Train classifier and save predictions
    predictions_df, accuracy = train_classifier(spark_df)
    
    # Detect anomalies and save anomalies
    anomalies_df = detect_anomalies(df)
    
    # Train clustering model and save results
    clustering_df = train_clustering(spark_df)
    
    return predictions_df, accuracy, anomalies_df, clustering_df

# Call ML operations before starting Dash app
perform_ml_operations()  # This will run the ML tasks and save results

# Define Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Continuous ML & Data Visualization Dashboard", style={"text-align": "center"}),

    daq.LEDDisplay(
        id="accuracy-display",
        label="Classifier Accuracy",
        value="0.00",
        color="#00FF00",
        backgroundColor="#000000",
        size=48,
    ),

    dcc.Interval(id="update-interval", interval=5000, n_intervals=0),  # Update every 5 seconds

    dcc.Tabs([
        dcc.Tab(label="Anomalies", children=[
            dcc.Graph(id="anomalies-graph"),
        ]),

        dcc.Tab(label="Clusters", children=[
            dcc.Graph(id="clusters-graph"),
        ]),

        dcc.Tab(label="Classification Results", children=[
            dcc.Graph(id="classification-graph"),
        ]),
    ]),
])

# Callback to update graphs and accuracy
@app.callback(
    [
        Output("anomalies-graph", "figure"),
        Output("clusters-graph", "figure"),
        Output("classification-graph", "figure"),
        Output("accuracy-display", "value"),
    ],
    [Input("update-interval", "n_intervals")],
)
def update_dashboard(n):
    # Load saved pickle files
    predictions_df = pd.read_pickle('predictions.pkl')
    anomalies_df = pd.read_pickle('anomalies.pkl')
    clustering_df = pd.read_pickle('clusters.pkl')

    # Get accuracy from the predictions
    accuracy = predictions_df['prediction'].mean()  # or use a separate accuracy variable saved earlier

    # Create figures for each tab
    anomalies_fig = px.scatter(
        anomalies_df, x="performance_score", y="job_satisfaction",
        color="anomaly", title="Detected Anomalies"
    )

    # Clustering visualization
    cluster_fig = px.scatter(
        clustering_df, x="performance_score", y="job_satisfaction",
        color="cluster", title="Clusters"
    )

    # Classification results visualization
    classification_fig = px.histogram(
        predictions_df, x="prediction", color="label", title="Classification Results"
    )

    return anomalies_fig, cluster_fig, classification_fig, str(round(accuracy, 2))

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
