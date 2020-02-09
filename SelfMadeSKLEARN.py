from sklearn import tree
import graphviz

features = [[0,0,1],[1,0,0],[0,1,1],[1,1,0],[1,0,1],[0,1,0],[0,0,1]]

labels = [1,1,0,0,1,0,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([1,1,1]))

tree.export_graphviz(clf,out_file='tree.dot') 
"""import warnings

def fxn():
	warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	fxn()"""