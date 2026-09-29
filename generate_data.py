import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

age_groups = np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], n,
                               p=[0.20, 0.30, 0.25, 0.15, 0.10])
occupations = np.random.choice(['Student', 'Salaried', 'Self-Employed', 'Retired', 'Business'], n,
                                p=[0.15, 0.35, 0.20, 0.10, 0.20])
cities = np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune'], n)

upi, card, atm, dep, wd, bill = [], [], [], [], [], []

for ag, occ in zip(age_groups, occupations):
    # Digital affinity by age
    digital_factor = {'18-25': 1.6, '26-35': 1.3, '36-45': 1.0, '46-55': 0.7, '56+': 0.4}[ag]
    atm_factor   = {'18-25': 0.4, '26-35': 0.7, '36-45': 1.0, '46-55': 1.3, '56+': 1.6}[ag]

    inc = {'Student': 0.5, 'Salaried': 1.0, 'Self-Employed': 1.2, 'Retired': 0.7, 'Business': 1.5}[occ]
    u = int(np.random.poisson(20 * digital_factor * inc))
    c = int(np.random.poisson(10 * digital_factor * inc))
    a = int(np.random.poisson(8  * atm_factor))
    d = round(np.random.normal(30000 * inc, 8000), 2)
    w = round(np.random.normal(20000 * inc, 6000), 2)
    b = int(np.random.poisson(5 * digital_factor))

    upi.append(max(u, 0)); card.append(max(c, 0)); atm.append(max(a, 0))
    dep.append(max(d, 1000)); wd.append(max(w, 500)); bill.append(max(b, 0))

anomaly_idx = np.random.choice(n, int(n * 0.02), replace=False)
for i in anomaly_idx:
    upi[i]  *= np.random.randint(4, 7)
    dep[i]  *= np.random.uniform(5, 10)

df = pd.DataFrame({
    'Customer_ID': [f'CUST{str(i+1).zfill(4)}' for i in range(n)],
    'Age_Group':   age_groups,
    'Occupation':  occupations,
    'City':        cities,
    'UPI_Count':   upi,
    'Card_Count':  card,
    'ATM_Count':   atm,
    'Deposits':    dep,
    'Withdrawals': wd,
    'Bill_Payments': bill
})

for col in ['UPI_Count', 'ATM_Count', 'Deposits']:
    df.loc[np.random.choice(n, 5, replace=False), col] = np.nan

df.to_csv('banking_transactions.csv', index=False)
print("Dataset saved: banking_transactions.csv")
print(df.head())
