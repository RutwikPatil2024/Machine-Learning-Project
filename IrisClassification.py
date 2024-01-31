from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


#loading iris dataset
iris=datasets.load_iris()


# features and labels
features=iris.data
labels=iris.target


#Training and classifier
clf=KNeighborsClassifier()    #classifier is ready
clf.fit(features,labels)      #classifier taking faatures and labels

predict=clf.predict([[1,1,1,1]])  #giving some data to classify
# 0- Iris - Setosa
# 1- Iris - Versicolour
# 2- Iris - Virginica
if predict==0:
    print("Setosa")
if predict==1:
    print("Setosa")
if predict==2:
    print("Setosa")