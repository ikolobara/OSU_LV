import numpy as np
import matplotlib.pyplot as plt

points = np.array([[1,1],[2,2],[3,2],[3,1],[1,1]])
plt.plot(points[:,0], points[:,1], 'g', linewidth=1, marker = "D", markersize=5)
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')
plt.axis([0,4,0,4])
plt.show()
