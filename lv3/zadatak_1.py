import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def taska(data):
    print(len(data))
    print(data.info())
    print(data.isnull().sum())
    print(data.duplicated().sum())

    data = data.dropna()
    data = data.drop_duplicates()
    data.reset_index(drop=True)

    object_columns = data.select_dtypes(include=["object"]).columns
    for col in object_columns:
        data[col] = data[col].astype("category") 

    print(data.info())


def taskb(data):
    print(data[["Model", "Make", "Fuel Consumption City (L/100km)"]].sort_values(ascending = False, by = ['Fuel Consumption City (L/100km)']).head(3))
    print(data[["Model", "Make", "Fuel Consumption City (L/100km)"]].sort_values(ascending = True, by = ['Fuel Consumption City (L/100km)']).head(3))


def taskc(data):
    filtered_data = data[(data["Engine Size (L)"] >= 2.5) & (data["Engine Size (L)"] <= 3.5)]
    print(len(filtered_data))
    print(filtered_data["CO2 Emissions (g/km)"].mean())


def taskd(data):
    filtered_data = data[data["Make"] == "Audi"]
    print(len(filtered_data))
    print(filtered_data[filtered_data["Cylinders"] == 4]["CO2 Emissions (g/km)"].mean())


def taske(data):
    grouped_data = data.groupby("Cylinders")
    print(grouped_data["Make"].count())
    print(grouped_data["CO2 Emissions (g/km)"].mean())


def taskf(data):
    print(data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].mean())
    print(data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].mean())
    print(data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].median())
    print(data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].median())


def taskg(data):
    print(data[(data["Fuel Type"] == "D") & (data["Cylinders"] == 4)].sort_values(ascending = False, by = ["Fuel Consumption City (L/100km)"]).head(1))
    
def taskh(data):
    print(len(data[data["Transmission"].str.startswith("M")]))


def taski(data):
    corr = data.corr(numeric_only=True)
    print(corr)
    sns.heatmap(corr, cmap='viridis')
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('lv3/data_C02_emission.csv')
    taskg(data)
