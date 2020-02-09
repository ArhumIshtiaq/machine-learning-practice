from sklearn import tree
from sklearn.metrics import accuracy_score

features = [[150,1],[185,0],[180,1],[140,0],[200,1],[199,0],[121,1],
      [220,0],[140,1],[175,0],[210,1],[99,0],[120,1],[120,0],[179,1]]
labels = [0,0,1,0,1,0,0,
          1,0,0,1,0,0,0,0]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

predictions = clf.predict([[180,0],[250,0],[180,1],[110,0],[195,1],[111,0]])
print(predictions)
y_test = [0,0,1,0,1,0]
print ("Accuracy (in %):", 100*accuracy_score(y_test, predictions))

tree.export_graphviz(clf,out_file='park.dot') 