# 💰 Smart Loan Approval Prediction System

An AI-powered Loan Approval Prediction System built with Streamlit, Machine Learning, and Interactive Analytics. The application helps financial institutions assess loan applications, analyze applicant risk, calculate affordability metrics, and perform bulk loan evaluations.

---

## 🚀 Features

### 📊 Interactive Dashboard

* Total Loan Applications
* Approved vs Rejected Applications
* Average Applicant Income
* Loan Amount Distribution Analysis
* Approval Ratio Visualization

### 🔍 Individual Loan Prediction

* Applicant Information Collection
* EMI Calculation
* Debt-to-Income (DTI) Analysis
* Disposable Income Assessment
* Credit Risk Evaluation
* AI-Based Loan Approval Prediction
* Hard Policy Validation Rules
* Confidence Score Generation

### 📂 Bulk Loan Scanner

* Batch Processing of Loan Applications
* CSV Upload Support
* JSON Upload Support
* SQL Template Support
* Bulk Prediction Engine
* Export Results in CSV, JSON, and SQL Formats

### 🛡 Risk Assessment Engine

* Credit Score Validation
* Employment Verification
* DTI Ratio Monitoring
* Affordability Analysis
* Collateral Coverage Check
* Dependents Impact Assessment

### 📈 Business Intelligence

* Automated Risk Categorization
* Trust Score Generation
* Approval Confidence Scores
* Audit Logs and Analytics

---

## 🏗 Project Structure

```bash
Loan-Approval-System/
│
├── app.py
├── loan_clean_data.csv
├── loan_model.pkl
├── loan_scaler.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* Python 3.x
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Plotly
* Pickle

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/loan-approval-prediction.git

cd loan-approval-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will open in browser:

```text
http://localhost:8501
```

---

## 📊 Input Features

| Feature            | Description                 |
| ------------------ | --------------------------- |
| Applicant Income   | Monthly applicant income    |
| Coapplicant Income | Monthly co-applicant income |
| Age                | Applicant age               |
| Dependents         | Number of dependents        |
| Credit Score       | Credit rating score         |
| Existing Loans     | Current active loans        |
| DTI Ratio          | Debt-to-Income ratio        |
| Savings            | Available savings           |
| Collateral Value   | Asset value pledged         |
| Loan Amount        | Requested loan amount       |
| Loan Term          | Loan tenure in months       |

---

## 🧮 EMI Calculation

The system automatically calculates EMI using:

```text
EMI = P × R × (1+R)^N
      ------------------
      (1+R)^N − 1
```

Where:

* P = Principal Loan Amount
* R = Monthly Interest Rate
* N = Loan Tenure (Months)

---

## 🚫 Hard Rejection Rules

Applications are automatically rejected if:

* Credit Score < 500
* DTI Ratio > 65%
* Negative Disposable Income
* Applicant is Unemployed
* Collateral Coverage < 50% of Loan Amount

These policy rules override the Machine Learning model prediction.

---

## 🤖 Machine Learning Workflow

1. User enters loan details.
2. Data is preprocessed using StandardScaler.
3. Features are passed to trained ML model.
4. Prediction probabilities are generated.
5. Hard policy checks are applied.
6. Final approval/rejection decision is returned.

---

## 📂 Bulk Processing Workflow

1. Download template file.
2. Upload CSV/JSON dataset.
3. Run AI Pipeline.
4. Review results.
5. Export processed predictions.

---

## 📸 Dashboard Modules

### Dashboard

* Loan Statistics
* Approval Insights
* Distribution Charts

### Loan Prediction

* EMI Estimation
* Risk Assessment
* AI Decision Engine

### Bulk Scanner

* Batch Evaluation
* Export Reports
* Audit Analytics

---

## 🔒 Smart Simulation Mode

If trained model files are unavailable:

```text
loan_model.pkl

```

the application automatically switches to Simulation Mode using fallback mock models.

---

## 📋 Requirements

```text
streamlit
pandas
numpy
scikit-learn
plotly
```

Install using:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```


---

## 👨‍💻 Author

Developed as an AI-powered Loan Risk Assessment and Approval System using Streamlit and Machine Learning.

---

## 📄 License

This project is licensed under the MIT License.
