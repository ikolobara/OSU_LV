import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50, 50))
white = np.ones((50, 50))

img = np.vstack((np.hstack((black, white)), np.hstack((white, black))))

plt.figure()
plt.imshow(img, cmap='gray')
plt.show()
