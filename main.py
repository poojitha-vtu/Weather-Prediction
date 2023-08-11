import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import os
weather = pd.read_csv(r"C:\Users\saipo\Downloads\weatherHistory.csv.zip")
weather
print(weather.head(10))
h=weather.groupby('Summary')["Apparent Temperature (C)"].mean().plot(kind='bar')
print(h)
plt.show()
weather_temp = weather[["Humidity","Apparent Temperature (C)"]]
print(weather_temp.head(12))
dummies = pd.get_dummies(weather["Summary"])
print(dummies.head(12))
weather_temp2 = pd.concat([weather_temp,dummies],axis=1)
print(weather_temp2.head(12))
Y = weather_temp["Apparent Temperature (C)"]
X = weather_temp2
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)
print(y_train.head(10))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
plt.scatter(X_test["Humidity"],y_predict,color='red')
plt.title("Temperature v/s humidity")
plt.xlabel("Humidity")
plt.ylabel("temperature")
plt.show()
plt.scatter(X_test["Humidity"],y_test,color="green")
#plt.plot(X_test,y_test)
#X_test.shape
plt.title("Temperature v/s humidity")
plt.xlabel("Humidity")
plt.ylabel("temperature")
plt.show()