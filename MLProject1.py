import pandas as pd
import numpy as np
from sklearn import svm
# import graphviz

df = pd.read_csv('Pokemon.csv')
d = np.asmatrix(df)
y = d[:,2]
x = d[:,5:11]

print(x)
print(y)