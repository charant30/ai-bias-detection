# Bias Detection Dashboard for Machine Learning Models

A lightweight web-based Streamlit application that allows users to train a simple machine learning model and evaluate whether the model produces biased outcomes across different demographic groups. The goal of this project is to help users move beyond standard model accuracy and better understand fairness in AI systems.

<img width="1917" height="1002" alt="image" src="https://github.com/user-attachments/assets/66ba10a4-3564-432c-8893-2a9e04ea7985" />


## Features
- **Data Input**: Upload any tabular CSV dataset or use the built-in synthetic demo dataset.
- **Model Configuration**: Select the target variable to predict and the sensitive attribute (e.g., gender, age group).
- **Automated Model Training**: Trains a basic Logistic Regression model automatically.
- **Fairness Metrics**: Evaluates potential bias by calculating Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR) per demographic group.
- **Visualizations**: Interactive and clear bar charts comparing performance metrics across groups.
- **Insight Generation**: Automated text summaries alerting you if performance disparities exceed a 10% fairness threshold.
- **Modern UI**: Clean and professional user interface powered by Streamlit and Ant Design components.

## Prerequisites
- Python 3.8+
- pip (Python package installer)

## Installation

1. Clone or download this repository.
2. Open a terminal/command prompt in the project directory (`ai-bias-detection`).
3. Create a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Once your virtual environment is activated and dependencies are installed, start the Streamlit server:
```bash
streamlit run app.py
```
The dashboard will automatically open in your default web browser (typically at `http://localhost:8501`).

## Usage Guide
1. **Choose Data Source**: On the sidebar, select **Upload CSV** or **Demo Dataset**.
   - If using the Demo Dataset, the app will automatically load a synthetic dataset with built-in gender bias for loan approvals.
2. **Configure Model Settings**: 
   - Select the **Target Column** you want the model to predict.
   - Select the **Sensitive Attribute** (the demographic group to analyze for bias).
3. **Train Model**: Click the "Train Model" button.
4. **Review Results**: Observe the overall accuracy, the bias evaluation charts, and read the generated insight summaries at the bottom to see if the model exhibits unfairness.

## About the Demo Dataset
The provided `demo_dataset.csv` is synthetically generated to simulate loan approvals. It intentionally includes bias where the 'Male' group has a significantly higher baseline chance of approval independent of merit based factors (like income and credit score) compared to the 'Female' group. This demonstrates how the model inherits hidden biases in historical data and how the dashboard detects them.

## Tech Stack
- **Frontend / UI**: [Streamlit](https://streamlit.io/), [streamlit-antd-components](https://nicedouble.github.io/streamlit-antd-components)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), NumPy
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/)
- **Visualizations**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
