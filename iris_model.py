from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# load dataset
iris = load_iris()

X = iris.data
y = iris.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create model
model = KNeighborsClassifier()

# train model
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
