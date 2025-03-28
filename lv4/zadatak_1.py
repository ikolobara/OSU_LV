import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

def task(data):
    wanted_columns = ['Engine Size (L)', 'Cylinders','Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']
    X = data[wanted_columns]
    y = data['CO2 Emissions (g/km)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    plt.scatter(X_train['Fuel Consumption City (L/100km)'], y_train, c='blue')
    plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test, c='red')
    plt.show()

    plt.subplot(1, 2, 1)
    plt.hist(X_train['Fuel Consumption City (L/100km)'])

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    plt.subplot(1, 2, 2)
    plt.hist(X_train_scaled[:, X_train.columns.get_loc('Fuel Consumption City (L/100km)')])
    plt.show()

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    print(lr.coef_) # theta1, theta2...
    print(lr.intercept_) # theta0

    y_pred = lr.predict(X_test_scaled)

    plt.scatter(y_test, y_pred)
    plt.show()

    print(mean_squared_error(y_test, y_pred))
    print(root_mean_squared_error(y_test, y_pred))
    print(mean_absolute_error(y_test, y_pred))
    print(mean_absolute_percentage_error(y_test, y_pred))
    print(r2_score(y_test, y_pred))

    #smanjenjem broja znacajki MSE, RMSE, MAE i MAPE se povecavaju, dok se R2 smanjuje sto znaci da nas model losije radi s manje znacajki.


if __name__ == "__main__":
    data = pd.read_csv('lv3/data_C02_emission.csv')
    task(data)