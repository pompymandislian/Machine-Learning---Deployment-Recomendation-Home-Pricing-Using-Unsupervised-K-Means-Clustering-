import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import requests
import numpy as np
import pandas as pd
import joblib
import time
import pickle
from sklearn.preprocessing import MinMaxScaler

yes_no = {1 : 'Yes', 0 : 'No'}
def format_func(option) :
    return yes_no[option]

#unpickle model and feature
kmeans_model = pickle.load(open("Kmeans.pkl", 'rb'))
minmax_feature = pickle.load(open("minmax.pkl", 'rb'))
data_deploy = pickle.load(open("data_fix.pkl", 'rb'))

st.set_page_config(
                    page_title= "Pricing Home Prediction", 
                    page_icon=":chart_with_upwards_trend:",
                    layout="wide",
                    initial_sidebar_state="expanded"
                   )

# load image
st.image('pacmann.png', width=400)

# function for get code url lottie
def load_lottieurl (url):
     r = requests.get(url)
     if r.status_code != 200:
          return None
     return r.json()

# load asset
lottie_asset=load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_LmW6VioIWc.json")

#--- Header Section ---
with st.container():
     st.subheader(" Hi I'm Bot Machine:robot_face:")
     st.title("Home Pricing Prediction")
     st.write("I can help you for search a few home according to your wishes and needs, you can input how many sum of bedroom, bathroom, garage, etc after you input values,then you can obtain recomendation home for you want!")
     st.markdown("**Note :** This is some important information.")
     left_note, right_note = st.columns(2)
     # Make list (Note)
     with left_note :
          st.write(" - max input for Building = 2672 (m2)")
          st.write("- max input for Garage = 10")
          st.write("- max input for Bedroom = 30")
          st.write("- max input for Bathroom = 31")
          st.write("- max input for land = 99506 (m2)")
     with right_note :
          st.write(" - max input for Installment_BCA = $ 455000000")
          st.write(" - max input for Installment_BNI = $ 119600000")
          st.write(" - max input for Installment_Mandiri = $ 292500000")
          st.write("- max input for Price = $ 97500000000")           
       
# input name user          
name_user = st.text_input('**Name User :**')
st.write(f'Hello Customer **{name_user}**')

with st.container():
     left_input, right_input = st.columns(2)
     with left_input :
          Price = st.slider('**Input Price** : ', min_value=0, max_value=97500000000, step=1000000)
          Building = st.slider('**Input Building (m2)** : ', min_value=0, max_value=2672, step=10) 
          Land = st.slider('**Land (m2)** :', min_value=0, max_value=99506, step=10) 
          Bedroom = st.number_input('**Input Bedroom** : ', step=1)
          Bathroom = st.number_input('**Input Bathroom** : ', step=1)
          Location_Kecamatan = st.selectbox('**select Location_Kecamatan :**', options= list(yes_no.keys()), format_func=format_func)
          Location_Kelurahan = st.selectbox('**select Location_Kelurahan :**', options= list(yes_no.keys()), format_func=format_func)
          Location_Kota = st.selectbox('**select Location_Kota :**', options= list(yes_no.keys()), format_func=format_func)
          
     with right_input :
          Installment_BCA = st.slider('**Input Installment_BCA** : ', min_value=0, max_value=455000000, step=1000000)
          Installment_Mandiri = st.slider('**Input Installment_Mandiri** : ', min_value=0, max_value=292500000, step=1000000)
          Installment_BNI = st.slider('**Input Installment_BNI** : ', min_value=0, max_value=119600000, step=1000000)
          Garage = st.number_input('**Input Garage** : ', step=1)
          Important_places = st.selectbox('**select Important_places :**', options= list(yes_no.keys()), format_func=format_func)
          No_important_places = st.selectbox('**select No_important_places :**', options= list(yes_no.keys()), format_func=format_func)
          Floor_Lantai_2 = st.selectbox('**select Floor Lantai 2 :**', options= list(yes_no.keys()), format_func=format_func)
          Floor_Lantai_3 = st.selectbox('**select Floor Lantai 3 :**', options= list(yes_no.keys()), format_func=format_func)
          Floor_Lantai_1 = st.selectbox('**select Floor Lantai 1 :**', options= list(yes_no.keys()), format_func=format_func)

# Make button prediction
if(st.button('Predict')):
     #url = "http://127.0.0.1:8000"
     # input data
     df_input = pd.DataFrame([[Installment_BNI, Garage, Price, Bedroom, Building, Installment_BCA, Land, Installment_Mandiri, Bathroom,
                           Important_places, No_important_places, Location_Kecamatan, Location_Kelurahan, Location_Kota, Floor_Lantai_2, Floor_Lantai_3,Floor_Lantai_1]],
                columns = ['Installment_BNI', 'Garage', 'Price', 'Bedroom', 'Building', 'Installment_BCA', 'Land', 'Installment_Mandiri', 'Bathroom',
                           'Important_places', 'No_important_places', 'Location_Kecamatan', 'Location_Kelurahan', 'Location_Kota', 'Floor_Lantai_2', 'Floor_Lantai_3','Floor_Lantai_1'] )
     
     #response = requests.post(f'{url}/prediction', json=df_input, timeout=8000)

     # animation     
     with st.spinner('wait for it...'):
          time.sleep(5)
     st.balloons()

     features= data_deploy.columns.to_list()

     cols =['Installment_BNI',
          'Garage',
          'Price',
          'Bedroom',
          'Building',
          'Installment_BCA',
          'Land',
          'Installment_Mandiri',
          'Bathroom',
          'Important_places',
          'No_important_places',
          'Location_Kecamatan',
          'Location_Kelurahan',
          'Location_Kota',
          'Floor_Lantai_2',
          'Floor_Lantai_3',
          'Floor_Lantai_1'] 

     df_input = df_input.transpose().reindex(features).transpose()
     minmax_feature.fit(df_input)
     inputdata = pd.DataFrame(minmax_feature.transform(df_input[cols]),columns = cols)
     result_prediction= kmeans_model.predict(inputdata)   

     # Output prediction
     if (result_prediction == 0) :
         result_prediction = "The result of recomendation cluster is : **0**"
     elif (result_prediction == 1) :
         result_prediction = "The result of recomendation cluster is : **1**"
     elif (result_prediction == 2) :
         result_prediction = "The result of recomendation cluster is : **2**"
     elif (result_prediction == 3) :
         result_prediction = "The result of recomendation cluster is : **3**"    
     elif (result_prediction == 4) :
         result_prediction = "The result of recomendation cluster is : **4**"
     elif (result_prediction == 5) :
         result_prediction = "The result of recomendation cluster is : **5**"
     elif (result_prediction == 6) :
         result_prediction = "The result of recomendation cluster is : **6**"    
     elif (result_prediction == 7) :
         result_prediction = "The result of recomendation cluster is : **7**"
     elif (result_prediction == 8) :
         result_prediction = "The result of recomendation cluster is : **8**"
     elif (result_prediction == 9) :
         result_prediction = "The result of recomendation cluster is : **9**"            

     st.success(result_prediction)

# Show lottie image
with st.container():
    st.write("----")
    left_col, right_col = st.columns(2)
    with right_col :
        st_lottie(lottie_asset, height = 300, key="coding")