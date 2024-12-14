import io
import base64
from flask import Flask, jsonify, render_template, request, Response
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Set non-interactive backend for matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Function to send email notification about high-risk employees
def send_email_notification(high_risk_employees):
    sender_email = 'rahultadepalli037@gmail.com'
    sender_password = 'tveo fxbz yvfq ltfa'
    recipient_email = 'mastanyeddu225@gmail.com'

    subject = "High-Risk Employee Attrition Alert"
    body = f"The following employees are identified as high-risk for attrition:\n\n" + "\n".join(map(str, high_risk_employees))

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print("Notification email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Data preprocessing function
def preprocess_employee_data(data, feature_columns=None):
    if 'Attrition' in data.columns:
        data['Attrition'] = LabelEncoder().fit_transform(data['Attrition'])

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    categorical_cols = data.select_dtypes(include='object').columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    if feature_columns is not None:
        for col in feature_columns:
            if col not in data.columns:
                data[col] = 0
        data = data[feature_columns]

    data = data.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, errors='ignore')

    return data

# Function to train the Random Forest model
def train_model():
    data = pd.read_csv('C:/hackathon/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    processed_data = preprocess_employee_data(data)

    X = processed_data.drop('Attrition', axis=1)
    y = processed_data['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns, data

# Train the model
model, feature_columns, raw_data = train_model()

# Flask route to serve frontend
@app.route('/')
def serve_frontend():
    return render_template('index.html')

# Flask route for about page
@app.route('/about')
def about():
    return render_template('about.html')

# Flask route for dashboard
@app.route('/dashboard', methods=['GET'])
def dashboard():
    processed_raw_data = preprocess_employee_data(raw_data, feature_columns=feature_columns)
    predictions = model.predict_proba(processed_raw_data)[:, 1]
    raw_data['Attrition_Prob'] = predictions

    high_risk = raw_data[raw_data['Attrition_Prob'] > 0.9]
    high_risk_numbers = high_risk['EmployeeNumber'].tolist()

    print("High-Risk Employees (Employee Numbers):")
    print(high_risk_numbers)

    if not high_risk.empty:
        send_email_notification(high_risk_numbers)

    avg_prob = predictions.mean()

    # Fetch the graph as an image from /attrition-trend-graph
    img_buffer = io.BytesIO()
    with app.test_client() as client:
        response = client.get('/attrition-trend-graph')
        img_buffer.write(response.data)
        img_buffer.seek(0)

    # Convert the image to base64
    graph_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{graph_base64}" alt="Attrition Trend Graph" />'

    # Construct the HTML response
    response = f"""
    <h1>Dashboard</h1>
    <p>Average Attrition Probability: {avg_prob:.2f}</p>
    <h2>High-Risk Employees</h2>
    <ul>
    """
    for index, row in high_risk.iterrows():
        response += f"<li>Employee Number: {row['EmployeeNumber']}, Department: {row['Department']}, Job Role: {row['JobRole']}, Attrition Probability: {row['Attrition_Prob']:.2f}</li>"
    response += "</ul>"
    response += "<h2>Attrition Trends</h2>"
    response += img_tag

    return Response(response, mimetype="text/html")

# Flask route for Satisfaction Analysis
@app.route('/Satisfication Analysis', methods=['POST','GET'])
def satisfication():
    data = pd.read_csv('C:/hackathon/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    employee_data = pd.DataFrame(data)

    # Group by 'Department' and calculate the average for each metric
    grouped_data = employee_data.groupby('Department').agg(
        avg_tenure=('TotalWorkingYears', 'mean'),
        avg_satisfaction=('JobSatisfaction', 'mean'),
        avg_performance=('PerformanceRating', 'mean')
    ).reset_index()

    # Generate the plot as a base64 image
    graph_base64 = plot_department_analysis(grouped_data)
    img_tag = f'<img src="data:image/png;base64,{graph_base64}" alt="Satisfaction Analysis Graph" />'

    # Construct the HTML response
    response = f"""
    <h1>Satisfaction Analysis</h1>
    <h2>Average Metrics by Department</h2>
    {img_tag}
    """

    return Response(response, mimetype="text/html")

# Function to generate plots for the grouped data
def plot_department_analysis(grouped_data):
    sns.set(style="whitegrid")

    # Create a plot with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot Average Tenure by Department
    sns.barplot(x='Department', y='avg_tenure', data=grouped_data, ax=axes[0], palette='viridis')
    axes[0].set_title('Average Tenure by Department')
    axes[0].set_ylabel('Average Tenure (Years)')

    # Plot Average Satisfaction by Department
    sns.barplot(x='Department', y='avg_satisfaction', data=grouped_data, ax=axes[1], palette='plasma')
    axes[1].set_title('Average Job Satisfaction by Department')
    axes[1].set_ylabel('Average Satisfaction')

    # Plot Average Performance by Department
    sns.barplot(x='Department', y='avg_performance', data=grouped_data, ax=axes[2], palette='inferno')
    axes[2].set_title('Average Performance Rating by Department')
    axes[2].set_ylabel('Average Performance Rating')

    # Rotate labels for readability
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save the plot to a BytesIO buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    # Encode the image as base64 to display in HTML
    graph_base64 = base64.b64encode(img.read()).decode('utf-8')
    return graph_base64

# Flask route for attrition trend graph
@app.route('/attrition-trend-graph')
def attrition_trend_graph():
    processed_raw_data = preprocess_employee_data(raw_data, feature_columns=feature_columns)
    predictions = model.predict_proba(processed_raw_data)[:, 1]
    raw_data['Attrition_Prob'] = predictions

    departments = raw_data['Department'].unique()
    trend = {}
    for dept in departments:
        dept_data = raw_data[raw_data['Department'] == dept]
        trend[dept] = dept_data['Attrition_Prob'].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(trend.keys(), trend.values(), color='skyblue')
    plt.title('Average Attrition Probability by Department')
    plt.xlabel('Department')
    plt.ylabel('Average Attrition Probability')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    return Response(img, mimetype='image/png')

# Flask route for exporting high-risk employee data
@app.route('/export-high-risk')
def export_high_risk():
    high_risk = raw_data[raw_data['Attrition_Prob'] > 0.9]
    csv_data = high_risk.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=high_risk_employees.csv"}
    )

if __name__ == '__main__':
    app.run(debug=True)