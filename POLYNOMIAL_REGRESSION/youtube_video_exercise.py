import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("insurance.csv")

df['sex'] = df['sex'].replace({
    'female': 1,
    'male': 2
})

df['smoker'] = df['smoker'].replace({
    'yes': 1,
    'no': 2
})

df['region'] = df['region'].replace({
    'southwest': 1,
    'southeast': 2,
    'northwest': 3,
    'northeast': 4
})

X = df.drop(columns=["charges"])
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


scaler = StandardScaler()

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

print("Before scaling:")
print(X_train.head())

print("\nAfter scaling:")
print(X_train_scaler[:5])