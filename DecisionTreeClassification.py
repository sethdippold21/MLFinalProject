import pandas as pd
import numpy as np
from sklearn import tree
import graphviz

df = pd.read_csv('Pokemon.csv')
d = np.asmatrix(df)
y = d[:,2]
x = d[:,5:10]
clf = tree.DecisionTreeClassifier()
clf.fit(x,y)
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 