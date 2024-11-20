import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

# Load pickle files
with open("department_metrics.pkl", "rb") as f:
    department_metrics_df = pickle.load(f)

with open("anomalies.pkl", "rb") as f:
    anomalies_df = pickle.load(f)

with open("rolling_avg.pkl", "rb") as f:
    rolling_avg_df = pickle.load(f)

with open("correlation.pkl", "rb") as f:
    correlation_result = pickle.load(f)

with open("role_distribution.pkl", "rb") as f:
    role_distribution = pickle.load(f)

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Employee Data Analysis Dashboard", style={"text-align": "center"}),

    dcc.Tabs([
        # Tab 1: Department Metrics
        dcc.Tab(label="Department Metrics", children=[
            html.Div([
                dcc.Graph(
                    id="department_metrics_graph",
                    figure=px.bar(department_metrics_df, x="department", y=["avg_performance_score", "avg_job_satisfaction"],
                                  barmode="group", title="Department-Wise Performance Metrics")
                )
            ])
        ]),

        # Tab 2: Anomalies (High Attrition Risk)
        dcc.Tab(label="Anomalies (High Attrition Risk)", children=[
            html.Div([
                dcc.Graph(
                    id="anomalies_graph",
                    figure=px.scatter(anomalies_df, x="timestamp", y="attrition_risk_numeric", color="department",
                                      title="Anomalies (High Attrition Risk)")
                )
            ])
        ]),

        # Tab 3: Rolling Averages (Productivity Index)
        dcc.Tab(label="Rolling Averages (Productivity Index)", children=[
            html.Div([
                dcc.Graph(
                    id="rolling_avg_graph",
                    figure=px.line(rolling_avg_df, x="timestamp", y="rolling_avg_productivity",
                                   title="Rolling Averages of Productivity Index")
                )
            ])
        ]),

        # Tab 4: Correlation Analysis
        dcc.Tab(label="Correlation Analysis", children=[
            html.Div([
                dcc.Graph(
                    id="correlation_graph",
                    figure=px.bar(x=["Productivity vs Attrition Risk"], y=[correlation_result["correlation"]],
                                  title="Correlation Between Productivity and Attrition Risk")
                )
            ])
        ]),

        # Tab 5: Role Distribution
        dcc.Tab(label="Role Distribution", children=[
            html.Div([
                dcc.Graph(
                    id="role_distribution_graph",
                    figure=px.pie(role_distribution, names="role", values="count",
                                  title="Distribution of Employee Roles")
                )
            ])
        ]),

        # Tab 6: Productivity Index Distribution
        dcc.Tab(label="Productivity Index Distribution", children=[
            html.Div([
                dcc.Graph(
                    id="productivity_distribution_graph",
                    figure=px.histogram(rolling_avg_df, x="productivity_index", nbins=50,
                                        title="Distribution of Productivity Index")
                )
            ])
        ]),

        # Tab 7: Total Work Hours by Department
        dcc.Tab(label="Work Hours by Department", children=[
            html.Div([
                dcc.Graph(
                    id="work_hours_graph",
                    figure=px.bar(department_metrics_df, x="department", y="total_work_hours",
                                  title="Total Work Hours by Department")
                )
            ])
        ])
    ])
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
