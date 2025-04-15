import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

def do_task(path):
    # ucitaj sliku
    img = Image.imread(path)

    # prikazi originalnu sliku
    # plt.figure()
    # plt.title("Originalna slika")
    # plt.imshow(img)
    # plt.tight_layout()
    # plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    print(len(np.unique(img_array, axis=0)))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    km = KMeans(n_clusters=5, init='k-means++', n_init=5, random_state=0)
    km.fit(img_array_aprox)
    labels = km.predict(img_array_aprox)

    centroids = km.cluster_centers_

    img_array_aprox = centroids[labels]
    img_array_aprox = np.reshape(img_array_aprox, (w,h,d))

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Approximation")
    plt.imshow(img_array_aprox)
    plt.axis("off")
    plt.show()

    inertias = []
    for i in range(2, 10):
        km = KMeans(n_clusters=i, init='k-means++', n_init=5, random_state=0)
        km.fit(img_array)
        inertias.append(km.inertia_)

    plt.figure()
    plt.plot(range(2, 10), inertias, marker='o')
    plt.show()

    for i in range(5):
        binary_mask = (labels == i).astype(np.uint8)
        binary_image = np.reshape(binary_mask, (w, h)) 

        plt.figure()
        plt.title(f'Cluster {i+1}')
        plt.imshow(binary_image, cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    for i in range(1,7):
        do_task(f'lv7/imgs/test_{i}.jpg')
