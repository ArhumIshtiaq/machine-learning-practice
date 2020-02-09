from scipy.spatial import distance
import numpy as np
from sklearn.metrics import accuracy_score

def euc(a,b):
	return distance.euclidean(a,b)

class NearestNeighbor():

	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		predictions= []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self,row):
		best_distance = euc(row, self.X_train[0])
		best_index = 0
		for i in range(1, len(self.X_train)):
			distance = euc(row, self.X_train[i])
			if best_distance > distance:
				best_distance = distance
				best_index = i
		return self.y_train[best_index]

classifier = NearestNeighbor()

classifier.fit([[150,1],[185,0],[180,1],[140,0],[200,1],[199,0],[121,1],
      [220,0],[140,1],[175,0],[210,1],[99,0],[120,1],[120,0],[179,1]],[0,0,1,0,1,0,0,
          1,0,0,1,0,0,0,0])

predictions = classifier.predict([[180,0],[250,0],[180,1],[110,0],[195,1],[111,0]])

print("Predictions:", predictions)

y_test = [0,0,1,0,1,0]

print ("Accuracy (in %):", 100*accuracy_score(y_test, predictions))
