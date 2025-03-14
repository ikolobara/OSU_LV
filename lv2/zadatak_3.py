import numpy as np
import matplotlib.pyplot as plt

def task3_a(img):
    img = img[:,:,0].copy()
    img += 10
    show_image(np.clip(img, 0, 255))
    

def task3_b(img):
    show_image(img[:, 160:320])


def task3_c(img):
    show_image(np.rot90(img, axes=(1,0)))


def task3_d(img):
    show_image(np.flip(img, axis=(1)))


def show_image(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == "__main__":
    img = plt.imread("lv2/road.jpg")
    task3_a(img)
    task3_b(img)
    task3_c(img)
    task3_d(img)
