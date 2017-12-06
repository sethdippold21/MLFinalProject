import pandas as pd
import numpy as np
from sklearn import naive_bayes
import graphviz

# read in the dataset
df = pd.read_csv('Pokemon.csv')
d = np.asmatrix(df)
y = np.array(d[:,2])
x = d[:,5:11]
size = x.shape[0]

averageTrainingAccuracy = 0
averageValidationAccuracy = 0
maximumValidationAccuracy = 0
for j in range(100):
    # Split the data into training and test sets
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

    # Train the Naive Bayes Classifier
    gnb = naive_bayes.GaussianNB()
    gnb.fit(trainx,trainy.ravel())

    correct = 0
    for i in range(len(trainy)):
        if trainy[i] == gnb.predict([trainx[i,:]]):
            correct = correct + 1
    averageTrainingAccuracy = averageTrainingAccuracy + (correct/len(trainy))

    correct = 0
    for i in range(len(validy)):
        if validy[i] == gnb.predict([validx[i,:]]):
                correct = correct + 1
    averageValidationAccuracy = averageValidationAccuracy + (correct/len(validy))
    if (correct/len(validy) > maximumValidationAccuracy):
        maximumValidationAccuracy = correct/len(validy)

print('Average Training Accuracy:', averageTrainingAccuracy, 'percent')
print('Average Validation Accuracy:', averageValidationAccuracy, 'percent')
print('Maximum:', maximumValidationAccuracy*100, 'percent')
