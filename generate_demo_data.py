import pandas as pd
import numpy as np

def generate_bias_data():
    np.random.seed(42)
    n = 1000
    
    # Features
    age = np.random.randint(18, 70, n)
    income = np.random.randint(20000, 150000, n)
    credit_score = np.random.randint(300, 850, n)
    
    # Sensitive Attribute: Gender (0=Male, 1=Female)
    gender = np.random.choice(['Male', 'Female'], n)
    
    # Target: Loan Approved (0=No, 1=Yes)
    # Introducing bias: Males have a higher baseline chance of approval independent of other factors
    approval_chance = np.zeros(n)
    
    for i in range(n):
        chance = 0.1 # Base chance
        
        # Merit based
        if credit_score[i] > 650:
            chance += 0.4
        if income[i] > 60000:
            chance += 0.2
            
        # Bias addition
        if gender[i] == 'Male':
            chance += 0.3
        
        approval_chance[i] = min(chance, 0.95)
        
    loan_approved = np.random.binomial(1, approval_chance)
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Credit_Score': credit_score,
        'Gender': gender,
        'Loan_Approved': loan_approved
    })
    
    # Map back target to easily understandable string for classification or just leave as is.
    df['Loan_Approved'] = df['Loan_Approved'].map({0: 'No', 1: 'Yes'})
    
    df.to_csv('demo_dataset.csv', index=False)
    print("demo_dataset.csv generated successfully.")

if __name__ == "__main__":
    generate_bias_data()
