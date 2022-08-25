import numpy as np
import pandas as pd

import gdown

from fastapi import FastAPI

url = 'https://drive.google.com/uc?id=1KI8LMBf95Gylq8heYcaKY2WXHYrI9wl9'

output = 'cos_sim_data.csv'

gdown.download(url,output, quiet = False)

cos_sim_data = pd.read_csv("cos_sim_data.csv")

map_df = pd.read_csv('./Data/Mapping.csv')

# map_df.set_index('Index',inplace=True)

app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "hello"
    }

@app.get("/Id/{ProductID}/{count}")
def GetSimilarProducts(ProductID: int, count: int = 5):

    prod_index = int(map_df[map_df['Prod_id'] == ProductID]['Index'].values)

    return cos_sim_data[cos_sim_data.columns[prod_index]].nlargest(count)
