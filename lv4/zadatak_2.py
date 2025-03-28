import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, max_error
import numpy as np


def task(data):
    ohe = OneHotEncoder(sparse_output=False)

    wanted_columns = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'Fuel Type']
    X = data[wanted_columns]
    
    encoded = ohe.fit_transform(X[['Fuel Type']])

    one_hot_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['Fuel Type']))

    X_encoded = pd.concat([X, one_hot_df], axis=1)
    
    X_encoded = X_encoded.drop(['Fuel Type'], axis=1)

    y = data['CO2 Emissions (g/km)']

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)
    lr = LinearRegression()

    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    print(mean_squared_error(y_test, y_pred))
    print(root_mean_squared_error(y_test, y_pred))
    print(mean_absolute_error(y_test, y_pred))
    print(mean_absolute_percentage_error(y_test, y_pred))
    print(r2_score(y_test, y_pred))

    print(max_error(y_test, y_pred))
    i = np.argmax(np.abs(y_pred - y_test))
    print(data['Make'][i], data['Model'][i])


if __name__ == '__main__':
    data = pd.read_csv('lv3/data_C02_emission.csv')
    task(data)
    