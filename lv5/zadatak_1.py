from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='coolwarm', marker='o')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', marker='x')
plt.show()

lr = LogisticRegression()
lr.fit(X_train, y_train)

theta_1, theta_2 = lr.coef_[0]
theta_0 = lr.intercept_[0]
print(f'{theta_0} + {theta_1} x1 + {theta_2} x2 = 0')
x1_values = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2_values = - (theta_0 + theta_1 * x1_values) / theta_2
plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='coolwarm', marker='o')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', marker='x')
plt.plot(x1_values, x2_values, 'k-')
plt.show()

y_pred = lr.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
ConfusionMatrixDisplay(confusion_matrix).plot()
plt.show()
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))

plt.scatter(X_test[y_test == y_pred, 0], X_test[y_test == y_pred, 1], color='green')
plt.scatter(X_test[y_test != y_pred, 0], X_test[y_test != y_pred, 1], color='black')
plt.show()
