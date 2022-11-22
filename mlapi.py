import pickle

import numpy as np
import pandas
import pandas as pd
import sklearn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

df = pd.read_csv('dataset (1).csv')
app = FastAPI()

origins = [
    "http://loaclhost:8000/new"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class User(BaseModel):
    disease1: str
    disease2: str
    veg: str


with open('food_rec.pkl', 'rb') as f:
    model = pickle.load(f)


@app.post("/new")
async def scoring_endpoint(item: User):
    d = {'anemia': 0, 'cancer': 0, 'diabeties': 0, 'eye_disease': 0, 'goitre': 0, 'heart_disease': 0, 'hypertension': 0,
         'kidney_disease': 0, 'obesity': 0, 'pregnancy': 0, 'rickets': 0, 'scurvy': 0, 'non-veg': 0, 'veg': 0}

    sampl = []
    for i in item:
        sampl.append(i[1])
    for i in sampl:

        d[i] = 1

    final_input = list(d.values())

    distnaces, indices = model.kneighbors([final_input])

    for i in list(indices):
        df_results = pd.DataFrame(columns=list(df.columns))
        df_results = df_results.append(df.loc[i])

    df_results = df_results.filter(
        ['Name', 'Nutrient', 'Veg_Non', 'Diet', 'Disease', 'description', 'VATTA', 'PITTA', 'KAPHA'])
    df_results = df_results.drop_duplicates(subset=['Name'])
    df_results = df_results.reset_index(drop=True)

    return df_results


if __name__ == '__main__':
    uvicorn.run(app,host="loaclhost" ,port=5000)
