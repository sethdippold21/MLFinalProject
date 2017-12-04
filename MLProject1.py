# import csv

# with open('Pokemon.csv', 'r') as f:
#   reader = csv.reader(f)
#   your_list = list(reader)

# print(your_list)

# import numpy

# numpy.loadtxt(open("Pokemon.csv", "rb"), delimiter=",", skiprows=1)

import pandas as pd 
df = pd.read_csv("Pokemon_Set1.csv")
print(df)