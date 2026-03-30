import pandas as pd
import ml_utils

try:
    df = pd.read_csv('demo_dataset.csv')
    print("Data loaded")
    gen_metrics, group_metrics = ml_utils.train_and_evaluate(df, 'Loan_Approved', 'Gender')
    print("General Metrics:", gen_metrics)
    print("Group Metrics:")
    print(group_metrics)
    print("Test Passed")
except Exception as e:
    print("Test Failed:", e)
