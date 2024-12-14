1. Features
Attrition Analysis:

Predicts the probability of employee attrition using a Random Forest model.
Identifies high-risk employees and sends email notifications.
Dashboard:

Displays average attrition probability.
Lists high-risk employees with detailed attributes.
Embeds attrition trend graphs.
Satisfaction Analysis:

Visualizes average tenure, job satisfaction, and performance by department.
Data Export:

Exports high-risk employee data as a CSV file.
Visualization:

Generates graphs for attrition trends and satisfaction analysis.
2. Routes
Frontend Pages
/

Serves the frontend index page (index.html).
/about

Serves the about page (about.html).
APIs and Analysis
Dashboard (/dashboard)

Method: GET
Generates a detailed dashboard with:
High-risk employees.
Attrition probability trend graphs.
Sends an email for high-risk employees if any are found.
Satisfaction Analysis (/Satisfication Analysis)

Method: POST, GET
Analyzes satisfaction metrics (tenure, satisfaction, performance) by department.
Returns an HTML response with graphs.
Attrition Trend Graph (/attrition-trend-graph)

Method: GET
Generates a bar chart showing average attrition probability by department.
Export High-Risk Employees (/export-high-risk)

Method: GET
Exports high-risk employee data (probability > 0.9) as a CSV file.
3. Key Components
3.1 Data Preprocessing
Function: preprocess_employee_data(data, feature_columns=None)
Cleans and processes employee data:
Encodes categorical variables.
Fills missing values.
Drops irrelevant columns (EmployeeNumber, Over18, etc.).
Ensures feature consistency using feature_columns.
3.2 Machine Learning
Model: Random Forest Classifier
Function: train_model()
Reads employee data (WA_Fn-UseC_-HR-Employee-Attrition.csv).
Preprocesses the dataset.
Splits data into training/testing sets.
Trains a Random Forest model.
Outputs the model, feature columns, and raw data.
3.3 Email Notification
Function: send_email_notification(high_risk_employees)
Sends an email to notify about high-risk employees.
Uses Gmail’s SMTP server for sending emails.
3.4 Visualization
Department Satisfaction Analysis:

Function: plot_department_analysis(grouped_data)
Generates bar plots for:
Average tenure.
Average job satisfaction.
Average performance rating.
Attrition Trend:

Function: /attrition-trend-graph
Creates a bar chart for average attrition probability by department.
4. Technologies and Libraries
Flask: Web framework for routing and server-side logic.
Machine Learning: scikit-learn (Random Forest model).
Data Processing: pandas, numpy.
Visualization: matplotlib, seaborn.
Email: smtplib, email.mime.
Templates: HTML templates rendered via Flask (render_template).
Others:
base64 for encoding graphs.
io for handling in-memory images.
5. Configuration
Email Credentials:

Sender email: rahultadepalli037@gmail.com.
SMTP server: smtp.gmail.com (Port: 587).
Note: Update sensitive credentials for production.
File Paths:

Employee data file: C:/hackathon/WA_Fn-UseC_-HR-Employee-Attrition.csv.
Flask Debug Mode:

Enabled (debug=True).
