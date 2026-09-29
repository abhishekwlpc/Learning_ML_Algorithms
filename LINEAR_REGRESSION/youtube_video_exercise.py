import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("salaries.csv")

x = df.iloc[:,:-1].values
print(x)

y = df.iloc[:,-1].values
print(y)

#plt.scatter(x,y)
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)
print("Training Set", x_train.shape)
print("Testing Set", x_test.shape)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

error = y_pred - y_test
print(error)

plt.scatter(x_train,y_train)
plt.scatter(x_test, y_pred, color='red')
plt.show()

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)