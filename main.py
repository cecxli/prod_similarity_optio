import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI

xls = pd.ExcelFile("./Data/ML Test Assignment - Similar Products Data.xlsx")
cat_df = pd.read_excel(xls,'Categories')
prod_df = pd.read_excel(xls,'Products')
prodsp_df = pd.read_excel(xls,'ProductSpecifications')

prodsp_df.drop_duplicates(inplace=True)

prod_df['Description'] = prod_df['Description'].map(str)

text_data = prod_df['Description']
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(text_data, show_progress_bar=True)

embeddings = pd.DataFrame(embeddings)

X = np.array(embeddings)

cos_sim_data = pd.DataFrame(cosine_similarity(X))

cos_sim_data.set_index(prod_df.ProductId,inplace=True)

map_df = pd.DataFrame()
map_df['Index'] = prod_df.index
map_df['Prod_id'] = prod_df['ProductId']


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
