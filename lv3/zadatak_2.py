import matplotlib.pyplot as plt
import pandas as pd


def taska(data):
    data["CO2 Emissions (g/km)"].plot(kind = "hist")
    plt.show()


def taskb(data):
    data['Fuel Type'] = data['Fuel Type'].astype("category")
    data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c="Fuel Type", colormap="inferno")
    plt.show()


def taskc(data):
    data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
    plt.show()


def taskd(data):
    data.groupby("Fuel Type").size().plot(kind='bar')
    plt.show()


def taske(data):
    data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().plot(kind='bar')
    plt.show()
    

if __name__ == "__main__":
    data = pd.read_csv('lv3/data_C02_emission.csv')
    taske(data)
