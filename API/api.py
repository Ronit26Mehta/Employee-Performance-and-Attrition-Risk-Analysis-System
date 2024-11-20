from flask import Flask, jsonify
from faker import Faker
import random
import threading
import time
import csv
import os
from datetime import datetime

app = Flask(__name__)
fake = Faker()

# File name for synthetic employee operation data
csv_file = "synthetic_employee_data.csv"

# Function to initialize CSV if it doesn't exist
def initialize_csv():
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp", "employee_id", "name", "department", "role", "age",
                "performance_score", "job_satisfaction", "promotion_status",
                "attrition_risk", "work_hours"
            ])

# Function to generate a single employee data record
def generate_employee_data():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    employee_id = random.randint(1000, 9999)
    name = fake.name()
    department = random.choice(["HR", "Engineering", "Sales", "Marketing", "Operations"])
    role = random.choice(["Manager", "Analyst", "Developer", "Consultant", "Executive"])
    age = random.randint(22, 60)
    performance_score = round(random.uniform(1, 5), 2)  # Scale: 1 to 5
    job_satisfaction = round(random.uniform(1, 10), 2)  # Scale: 1 to 10
    promotion_status = random.choice(["Promoted", "Not Promoted"])
    attrition_risk = random.choice(["Low", "Medium", "High"])
    work_hours = random.randint(30, 60)

    # Introduce data variation (e.g., high attrition risk with low satisfaction)
    if job_satisfaction < 3:
        attrition_risk = "High"

    return {
        "timestamp": timestamp,
        "employee_id": employee_id,
        "name": name,
        "department": department,
        "role": role,
        "age": age,
        "performance_score": performance_score,
        "job_satisfaction": job_satisfaction,
        "promotion_status": promotion_status,
        "attrition_risk": attrition_risk,
        "work_hours": work_hours
    }

# Background task to append generated data to CSV every second
def generate_data_continuously():
    while True:
        employee_data = generate_employee_data()
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(employee_data.values())
        time.sleep(1)

# API route to get recent data from CSV
@app.route('/recent_data', methods=['GET'])
def recent_data():
    employee_data = []
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                employee_data.append(row)
    return jsonify(employee_data[-10:])

# API route to fetch all data from CSV
@app.route('/fetch_all_data', methods=['GET'])
def fetch_all_data():
    employee_data = []
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                employee_data.append(row)
    return jsonify(employee_data)  # Return all records as JSON

if __name__ == '__main__':
    # Initialize CSV and start background data generation
    initialize_csv()
    threading.Thread(target=generate_data_continuously, daemon=True).start()
    app.run(port=5000)
