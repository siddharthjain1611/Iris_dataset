from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris=load_iris()

# splitting the features and the labels
features=iris.data
labels=iris.target

labels_names = ['I.setosa', 'I.versicolor', 'I.virginica']
colors=['blue', 'red', 'green']

# plot-1 between sepal length and sepal width
for i in range(len(colors)):
    px=features[:,0][labels==i]
    py=features[:,1][labels==i]
    plt.scatter(px, py, c=colors[i])
plt.legend(labels_names)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


# plot-2 between petal length and petal width
for i in range(len(colors)):
    px=features[:,1][labels==i]
    py=features[:,2][labels==i]
    plt.scatter(px, py, c=colors[i])
plt.legend(labels_names)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

# Estimating two principle components using PCA
est=PCA(n_components=2)
x_pca=est.fit_transform(features)

colors=['black', 'orange', 'pink']
for i in range(len(colors)):
    px=x_pca[:,0][labels==i]
    py=x_pca[:,1][labels==i]
    plt.scatter(px, py, c=colors[i])
plt.legend(labels_names)
plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')
plt.show()

# splitting testing ab=nd training data
x_train, x_test, y_train, y_test = train_test_split(
    x_pca, labels, test_size=0.4, random_state=33)

# training the SVM classifier
clf=SVC()
clf.fit(x_train, y_train)

# predicting the results
pred = clf.predict(x_test)

# generate evaluation report
from sklearn import metrics
print(metrics.classification_report(
    y_test, pred, target_names=labels_names))