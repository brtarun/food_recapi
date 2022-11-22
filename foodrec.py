# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:05:05 2022

@author: brpktk
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:11:27 2022

@author: brpktk
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('dataset (1).csv')
class Recommender:
    
    def __init__(self):
        self.df = pd.read_csv('dataset (1).csv')
    
    def get_features(self):
        #getting dummies of dataset
        #nutrient_dummies = self.df.Nutrient.str.get_dummies()
        disease_dummies = self.df.Disease.str.get_dummies(sep=' ')
        #diet_dummies = self.df.Diet.str.get_dummies(sep=' ')
        veg=self.df.Veg_Non.str.get_dummies()
        feature_df = pd.concat([disease_dummies,veg],axis=1)
     
        return feature_df
    
    def k_neighbor(self,inputs):
        
        feature_df = self.get_features()
        
        #initializing model with k=20 neighbors
        model = NearestNeighbors(n_neighbors=40,algorithm='ball_tree')
        
        # fitting model with dataset features
        model.fit(feature_df)
        
        df_results = pd.DataFrame(columns=list(self.df.columns))
        
      
        # getting distance and indices for k nearest neighbor
        distnaces , indices = model.kneighbors(inputs)

        for i in list(indices):
            df_results = df_results.append(self.df.loc[i])
                
        df_results = df_results.filter(['Name','Nutrient','Veg_Non','Diet','Disease','description','VATTA'])
        df_results = df_results.drop_duplicates(subset=['Name'])
        df_results = df_results.reset_index(drop=True)
        return df_results
ob = Recommender()
data = ob.get_features()

total_features = data.columns
d = dict()
for i in total_features:
    d[i]= 0
print(d)
sample_input = ['diabeties','anemia','veg',]

for i in sample_input:
    
    d[i] = 1

final_input = list(d.values())
results = ob.k_neighbor([final_input]) # pass 2d array []
print(results)