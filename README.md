

---

# Employee Performance and Attrition Risk Analysis System

## Overview
The **Employee Performance and Attrition Risk Analysis System** is a comprehensive tool designed to assist organizations in evaluating employee performance, identifying attrition risks, and making data-driven decisions. This system leverages machine learning, data visualization, and synthetic data generation to provide actionable insights.

## Features
- **Synthetic Data Generation**: Create realistic employee performance datasets using Faker.
- **API Integration**: A Flask-based API for managing and accessing synthetic employee data.
- **Interactive Dashboards**: Visualize key performance metrics, anomalies, and trends using Dash and Plotly.
- **Machine Learning Models**:
  - **Clustering**: Identify patterns and employee groups using K-Means.
  - **Anomaly Detection**: Detect outliers with Isolation Forests.
  - **Attrition Prediction**: Predict employee churn using Random Forest models.
- **Pre-generated Insights**: Includes pre-saved metrics like anomalies, rolling averages, and department-wise KPIs.

## Project Structure
```
Employee-Performance-and-Attrition-Risk-Analysis-System-main/
├── .gitignore                 # Ignored files for version control
├── README.md                  # Documentation (current file)
├── requirements.txt           # Dependencies
├── API/                       # API for data operations
│   └── api.py
├── Dashboard/                 # Dash-based visualization
│   └── dashboard.py
├── Data/                      # Input dataset
│   └── synthetic_employee_data.csv
├── Model/                     # Machine learning workflows
│   └── ml.py
├── output/                    # Pre-saved insights and metrics
│   ├── anomalies.pkl
│   ├── clusters.pkl
│   ├── correlation.pkl
│   ├── department_metrics.pkl
│   ├── predictions.pkl
│   ├── role_distribution.pkl
│   └── rolling_avg.pkl
└── transformer/               # Data transformation utilities
    └── transform.py
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Employee-Performance-and-Attrition-Risk-Analysis-System-main
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up additional tools:
   - Ensure **Python 3.8+** is installed.
   - Install PySpark for large-scale machine learning operations.

## Usage
### API
Run the Flask API to manage employee data:
```bash
python API/api.py
```
Access endpoints for:
- Generating new employee data.
- Exporting datasets for analysis.

### Dashboard
Start the dashboard to visualize data:
```bash
python Dashboard/dashboard.py
```
Explore:
- Performance trends.
- Anomalies and correlation metrics.
- Department-wise KPIs.

### Machine Learning
Use the ML script for advanced analyses:
```bash
python Model/ml.py
```
Capabilities include:
- Clustering and visualization.
- Anomaly detection.
- Attrition prediction.

## Technical Details
- **Data Sources**: Synthetic datasets generated with Faker.
- **Machine Learning**:
  - Clustering: K-Means with PySpark.
  - Anomaly Detection: Isolation Forest.
  - Prediction: Random Forest models.
- **Visualization**: Dash framework with Plotly for interactive visuals.
## outputs:
  1. Transformed Data using pyspark:

     ![Screenshot 2024-11-20 112252](https://github.com/user-attachments/assets/717494f9-a055-400d-b700-e0cd2dcc7176)

     ![Screenshot 2024-11-20 112005](https://github.com/user-attachments/assets/c162dab0-624a-4f5b-ad13-d1edc1bbba4c)

     ![Screenshot 2024-11-20 112152](https://github.com/user-attachments/assets/2a21d15a-fdcf-4ca6-8b2b-07420db81dc9)

     ![Screenshot 2024-11-20 112231](https://github.com/user-attachments/assets/d2451553-40ee-4a78-8633-01e36890a602)
    
2. dashboard 

    ![Screenshot 2024-11-20 112523](https://github.com/user-attachments/assets/38f2258a-a3a2-4c37-8ce0-1dab1672f8a8)

    ![Screenshot 2024-11-20 112454](https://github.com/user-attachments/assets/9b59c778-9d64-41b7-9c3b-60a4857842f3)


   ![Screenshot 2024-11-20 112415](https://github.com/user-attachments/assets/acf975b2-12d6-4f7a-b145-4cc4bf75773c)
   



## Future Enhancements
- Extend the dataset to include real-world scenarios.
- Integrate more advanced models for attrition prediction.
- Deploy the system on a cloud platform for scalability.

## License
This project is licensed under the MIT License.

---
