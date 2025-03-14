import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("lv2/data.csv", dtype=float, delimiter=',')
data = data[1:]
print(len(data))

plt.scatter(data[:, 1], data[:, 2], marker = '.')
plt.show()

plt.scatter(data[::50, 1], data[::50, 2], marker = '.')
plt.show()

print(data[:,1].min())
print(data[:,1].max())
print(data[:,1].mean())

indm = (data[:,0] == 1)
indf = (data[:,0] == 0)

print(data[indm,1].min())
print(data[indm,1].max())
print(data[indm,1].mean())

print(data[indf,1].min())
print(data[indf,1].max())
print(data[indf,1].mean())

