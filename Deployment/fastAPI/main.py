from sklearn.cluster import KMeans
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Open data fix
with open('data_fix.pkl', 'rb') as file:
      
    # Call load method to deserialze
    data_fix = pickle.load(file)

app = FastAPI()

class dtype(BaseModel):

    Installment_BCA : float
    Bathroom : float
    Installment_Mandiri : float
    Building : float
    Price : float
    Garage : float
    Bedroom : float
    Bathroom : float
    Installment_BNI : float
    Land : float
    Installment_BNI : float
    Location_Kecamatan : float
    Location_Kelurahan : float
    Location_Kota : float
    Floor_Lantai_2 : float
    Floor_Lantai_3 : float
    Floor_Lantai_1 : float
    Important_places : float
    No_important_places : float

@app.get('/')

def index():

    return {'message': 'This is the homepage of the API '}

origins = [
    "http://localhost:80",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)    

@app.post('/prediction')

async def get_data(data: dtype):
    try:

        install_bca = data_fix['Installment_BCA']
        install_mandiri = data_fix['Installment_Mandiri'] 
        build = data_fix['Building'] 
        price = data_fix['Price'] 
        bedroom = data_fix['Bedroom']
        bathroom = data_fix['Bathroom']  
        install_bni = data_fix['Installment_BNI'] 
        land = data_fix['Land'] 
        garage = data_fix['Garage'] 
        loc_kec = data_fix['Location_Kecamatan']
        loc_kel = data_fix['Location_Kelurahan'] 
        loc_kota = data_fix['Location_Kota'] 
        fl_2 = data_fix['Floor_Lantai_2'] 
        fl_3 = data_fix['Floor_Lantai_3']
        fl_1 = data_fix['Floor_Lantai_1']
        importplace = data_fix['Important_places']
        unimportplace = data_fix['No_important_places']

        input_list = [install_bca, install_mandiri, build, price, garage, bedroom, loc_kec, loc_kel, loc_kota, 
                      fl_2, fl_3, fl_1, bathroom, install_bni, land, importplace, unimportplace]
        
        kmeans = KMeans(n_clusters=7) # input cluster
        kmeans.fit(input_list)

        prediction = kmeans.predict(input_list).tolist()[0]
    except ZeroDivisionError:
        return {"error": "Cannot divide by zero"}
    else:
        return {'prediction Cluster is': prediction}

