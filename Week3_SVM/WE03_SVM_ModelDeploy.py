#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
#import os
#exit(os.getcwd())

Lawnmover_model = pickle.load(open("C:/Users/vvenk/Box/MS BAIS Venkatesh/DSP/week3/pickle.csv", "rb"))

print("\n*****************************************************")
print("* Lawnmover Ownership Prediction Model *")
print("*****************************************************\n")
Income = float(input("Enter the Income: "))
Lot_Size= float(input("Enter the Lotsize: "))
df = pd.DataFrame({'Income': [Income], 'Lot_Size': [Lot_Size]})
result = Lawnmover_model.predict(df)
probability = Lawnmover_model.predict_proba(df)
Ownership = ('Owner', 'Nonowner')
print(f"\nThe Lawnmover Ownership Prediction Model indicates probability of Ownership at {probability[0][1]:.4f}, therefore it is indicated as: {Ownership[result[0]]}.\n")


# In[ ]:




