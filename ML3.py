from scipy.spatial import distance
from sklearn import datasets
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)


def euc(a, b):
	return distance.euclidean(a, b)

class ScrappyKNN():

	def fit(self, X_train, y_train):
		self.X_train = X_train;
		self.y_train = y_train;

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self, row):
		best_distance = euc(row, self.X_train[0])
		best_index = 0
		for i in range(1, len(self.X_train)):
			distance = euc(row, self.X_train[i])
			if distance < best_distance:
				best_distance = distance
				best_index = i
		return self.y_train[best_index]

my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print ("Accuracy (in %):", 100*accuracy_score(y_test, predictions))