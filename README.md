# ml_project
Customer Churn Prediction Machine Learning Project
Customer Churn Prediction - Telecom Industry

Overview :
This repository contains the codebase for a machine learning project aimed at predicting customer churn within the telecom industry. The project uses data-driven analysis and predictive modeling to identify customers who are likely to discontinue their service. By understanding the key factors influencing churn, this model helps telecom providers reduce customer loss and improve overall retention strategies.

Objective:
The goal of this project is to build a robust, interpretable model capable of forecasting whether a customer will churn based on various demographic, service usage, and account-related factors. The predictive insights can assist in proactive retention campaigns and strategic business decisions.

Key Features

1. Data Preprocessing:
Automatic handling of missing values, outlier detection, feature encoding, and data normalization.

2. Exploratory Data Analysis (EDA):
Visualizations and statistical summaries to uncover trends, correlations, and patterns related to churn behavior.

3. Feature Engineering:
Creation of derived attributes such as tenure categories, service utilization ratios, and payment consistency indicators.

4. Model Training and Evaluation:
Includes implementations of multiple supervised learning algorithms such as Logistic Regression, Random Forest, XGBoost, and Gradient Boosting.
Models are compared based on accuracy, precision, recall, F1-score, and ROC-AUC performance metrics.

5. Hyperparameter Optimization:
Uses Grid Search and Random Search techniques to fine-tune hyperparameters and improve predictive accuracy.

6. Model Interpretability:
Feature importance analysis and SHAP values to understand the most influential predictors behind customer churn.

7. Reproducibility:
All notebooks and scripts are modular and version-controlled for consistent experiment tracking.

Technologies Used
1. Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, SHAP)

2. Jupyter Notebook

3. Machine Learning & Data Science Libraries

4. Git for version control

Project Structure
churn-prediction/
│
├── data/                   # Dataset (raw and processed)
├── notebooks/              # Jupyter notebooks for data exploration and modeling
├── src/                    # Python scripts for data preprocessing and model training
├── models/                 # Saved model files
├── results/                # Evaluation metrics and plots
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

How to Use the Code
1. Clone this Repository
   git clone https://github.com/pranavgodge9-collab/ml_project.git
   cd churn-prediction
   
2. Install Dependencies
   pip install -r requirements.txt
   
3. Prepare the Data
   Place your telecom dataset in the data/ directory. Update file paths in the preprocessing script if required.

4. Run Data Preprocessing
   python src/preprocess_data.py

5. Train the Model
   python src/train_model.py

6. Evaluate the Model
   Evaluation results will be saved automatically under the results/ directory including performance metrics and visual plots.

7. Predict Churn for New Customers
   python src/predict.py --input data/new_customers.csv --output results/predictions.csv


