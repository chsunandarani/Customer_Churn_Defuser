# Customer Churn Defuser

**AI-Powered Customer Retention and Churn Prediction System**

Customer Churn Defuser is a Machine Learning-powered web application that predicts customers who are likely to leave a subscription-based business. The application assigns a churn risk score and provides personalized retention recommendations, helping businesses take proactive actions before customers churn.

---

## 🚀 Features

- 📊 Interactive Dashboard
- 🤖 Machine Learning Churn Prediction
- 📈 Customer Risk Score (0–100%)
- 💡 Smart Retention Recommendations
- 📂 Bulk CSV Upload and Prediction
- 📉 Interactive Analytics using Plotly
- 📊 Feature Importance Visualization
- 📱 Responsive Modern UI
- ☁️ Deployment Ready (Render)

---

## 🛠 Tech Stack

### Frontend
- HTML5
- CSS3
- Bootstrap 5
- JavaScript
- Chart.js
- Plotly.js

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

### Data Processing
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn
- Plotly

### Deployment
- Render

---

# 📂 Project Structure

```text
Customer_Churn_Defuser/
│
├── app.py
├── requirements.txt
├── Procfile
├── render.yaml
├── README.md
├── churn_model.pkl
├── scaler.pkl
├── dataset_generator.py
├── train_model.py
│
├── dataset/
│   └── customer_churn_dataset.csv
│
├── models/
│   └── model_metrics.json
│
├── notebooks/
│   ├── 01_Dataset_Generation.ipynb
│   └── 02_EDA_and_Model_Training.ipynb
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── prediction.html
│   ├── upload.html
│   ├── results.html
│   └── analytics.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
```

---

# 📊 Dataset

The project generates a synthetic dataset with **10,000 customer records**.

### Dataset Features

- Customer ID
- Age
- Gender
- Subscription Type
- Monthly Fee
- Subscription Months
- Login Frequency
- Session Duration
- Support Tickets
- Payment Failures
- Last Active Days
- Customer Satisfaction
- Feature Usage Score
- Referrals
- Engagement Score
- Churn (Target)

Target Variable

- **0 → Active Customer**
- **1 → Churned Customer**

---

# 📈 Machine Learning Pipeline

The application automatically:

- Generates dataset
- Performs Exploratory Data Analysis (EDA)
- Trains multiple ML models
- Compares model performance
- Selects the best model
- Saves the trained model

Algorithms Used

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

# 📊 Dashboard Features

### Dashboard

Displays:

- Total Customers
- Active Customers
- Churn Rate
- High Risk Customers
- Medium Risk Customers
- Low Risk Customers

---

### Customer Prediction

Input customer information and receive:

- Churn Prediction
- Risk Score
- Risk Level
- Retention Recommendation

---

### Bulk Prediction

Upload a CSV file containing customer data.

The application predicts churn for every customer and allows downloading the prediction results.

---

### Analytics

Interactive charts include:

- Churn Distribution
- Risk Distribution
- Feature Importance
- Customer Segmentation
- Correlation Heatmap

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/chsunandarani/Customer_Churn_Defuser.git
```

Navigate into the project

```bash
cd Customer_Churn_Defuser
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Dataset Generator

```bash
python dataset_generator.py
```

---

# ▶️ Train Machine Learning Model

```bash
python train_model.py
```

---

# ▶️ Run the Flask Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

---

# 🌐 Deployment

This project is deployment-ready on **Render**.

Build Command

```bash
pip install -r requirements.txt && python dataset_generator.py && python train_model.py
```

Start Command

```bash
gunicorn app:app
```

---

# 📌 Future Improvements

- User Authentication
- Email Notification System
- Live Customer Database
- Real-Time Predictions
- REST API Integration
- Docker Support
- Cloud Database Integration
- Admin Panel

---

# 👨‍💻 Author

**Sunanda Rani Chowdary Chapa**

GitHub: https://github.com/chsunandarani

---

# ⭐ If you found this project useful, consider giving it a Star on GitHub!