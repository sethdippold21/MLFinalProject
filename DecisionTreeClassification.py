import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
from sklearn.datasets import load_iris

# read in the dataset
df = pd.read_csv('Pokemon.csv')
d = np.asmatrix(df)
y = np.array(d[:,2])
x = d[:,5:11]
size = x.shape[0]

# Split the data for k-fold cross validation using 5 sections
shuffled_x = np.empty(x.shape, dtype=x.dtype)
shuffled_y = np.empty(y.shape, dtype=y.dtype)
permutation = np.random.permutation(size)
for old_index, new_index in enumerate(permutation):
    shuffled_x[new_index] = x[old_index]
    shuffled_y[new_index] = y[old_index]
x = shuffled_x
y = shuffled_y
numInPart = int(size/5)
trainx,validx = x[:4*numInPart,:],x[4*numInPart:,:]
trainy,validy = y[:4*numInPart,:],y[4*numInPart:,:]

# Run each fold
clf = tree.DecisionTreeClassifier()
clf.fit(trainx,trainy)
correct = 0
for i in range(len(validy)):
    if validy[i] == clf.predict([validx[i,:]]):
            correct = correct + 1

print(correct/len(validy))

data_names = ['HP','Attack','Defense','Sp. Attack','Sp. Defense','Speed']

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=data_names,
                                class_names=np.unique(y),
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data) 
graph.render("DecisionTreeClassifier")