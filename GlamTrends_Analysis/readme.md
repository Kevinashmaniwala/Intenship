# 🛍️ GlamTrends: End-to-End Women's Clothing AI & Analytics

## 📌 Project Overview

This project provides a 360-degree analysis of the **Women’s Clothing E-Commerce Reviews** dataset. It combines advanced Natural Language Processing (NLP), Machine Learning, and Interactive Dashboards to help stakeholders understand customer sentiment and product performance.

## 🚀 Key Features

### 1. Interactive AI Web Studio (`App.py`)

Built using **Streamlit**, this custom-built "Enterprise AI Studio" allows for:

* **Dynamic Data Connection:** Connects to SQLite databases or CSV sources.
* **AI-Powered Sentiment Analysis:** Utilizes the `transformers` pipeline for deep-text emotional analysis of customer reviews.
* **Auto-ML Training:** Users can choose between **Random Forest** and **XGBoost**, tune hyperparameters (Learning Rate, Depth), and train models directly from the UI.
* **Feature Interpretability:** Displays Feature Impact charts and Word Clouds to show exactly what drives customer satisfaction.

### 2. Advanced Machine Learning Pipeline

* **Text Processing:** Implemented `TfidfVectorizer` and `CountVectorizer` for N-gram (Bigram) analysis.
* **Model Performance:** Utilized `StandardScaler` for normalization and `PCA` (Principal Component Analysis) for dimensionality reduction.
* **Linear Execution Logic:** As per professional standards, the backend code follows a clean, sequential flow for maximum transparency and debugging ease.

### 3. Professional Business Intelligence

* **Power BI Integration:** The project is supported by a premium **Navy Blue & Gold** themed dashboard (refer to the `pbix` file).
* **Key KPIs:** Tracking Total Reviews, Average Ratings, Recommended Rates, and Department-wise performance.
* **Demographic Insights:** Analysis of how "Age" and "Division" influence buying patterns.

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python-based Web Framework)
* **Languages:** Python (Pandas, NumPy, Scikit-learn)
* **AI/ML:** XGBoost, Transformers (HuggingFace), Matplotlib, Seaborn, Plotly
* **Data Storage:** SQLite, CSV
* **Visualization:** Power BI Desktop

## 📁 Repository Structure

* `App.py`: The main Streamlit application script.
* `06-01-26-Charts.ipynb`: Data visualization and exploratory data analysis (EDA).
* `23-01-26.ipynb`: NLP, semantic link analysis, and model training.
* `final_model.pkl`: The saved, trained model for deployment.

---

### **How to Run the Application**

1. **Clone the Repo:**
`git clone https://github.com/your-username/Internship.git`
2. **Install Requirements:**
`pip install streamlit pandas scikit-learn xgboost transformers plotly`
3. **Launch the App:**
`streamlit run App.py`

---
