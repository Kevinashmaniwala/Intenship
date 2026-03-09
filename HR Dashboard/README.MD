# 📊 Strategic HR Workforce Analytics Dashboard

## 🚀 Project Overview

This project focuses on analyzing HR data to bridge the gap between employee professional development and organizational performance. Using a dataset of over 3,000 records, this dashboard provides high-level insights into **Training ROI**, **Workforce Demographics**, and **Employee Sentiment** to drive data-backed executive decisions.

## 🛠️ Key Features & Industry Insights

* **Training ROI & Success:** Moving beyond basic counts to calculate the actual success rate of training programs.
* **Executive KPIs:** Focuses on high-impact metrics like **Training Success Rate %** and **High Engagement %**.
* **Time-Series Analysis:** Visualizes the trend of training costs over the years, categorized by Internal and External types.
* **Departmental Benchmarking:** Compares training success rates across different departments to identify high-performing business units.
* **Sentiment Correlation:** Analyzes the relationship between employee satisfaction scores and their actual engagement levels.

## 📊 Dashboard Visuals

* **Executive Scorecard:** Real-time tracking of Training Success Rate, Average Age, Total Training Cost, and Engagement %.
* **Demographics:** Breakdown of gender and racial diversity within the workforce.
* **Performance Distribution:** Analysis of employee performance ratings from 'Needs Improvement' to 'Exceeds Expectations'.
* **Operational Deep-dive:** Interactive scatter plots and bar charts for department-wise performance.

## 🧰 Tech Stack

* **Visualization Tool:** Power BI Desktop.
* **Analytics Language:** DAX (Data Analysis Expressions).
* **Design Theme:** Executive Navy Blue & Gold (Industry Standard for Premium Dashboards).
* **Data Processing:** Cleaned and transformed 3,000+ records to ensure data integrity.

## 📈 Advanced DAX Measures

```dax
Training Success Rate % = 
DIVIDE(
    CALCULATE(COUNT(HR_Data[Employee ID]), HR_Data[Training Outcome] IN {"Passed", "Completed"}),
    COUNT(HR_Data[Employee ID]),
    0
)

High Engagement % = 
DIVIDE(
    CALCULATE(COUNT(HR_Data[Employee ID]), HR_Data[Engagement Score] >= 4),
    COUNT(HR_Data[Employee ID]),
    0
)

-------
