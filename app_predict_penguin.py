
import streamlit as st 
from streamlit_option_menu import option_menu
import plotly.graph_objects as px
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("My Penguin Species Predition <3")

tab1, tab2, tab3 = st.tabs(["Penquin Prediction", "Evaluation", "About"])

model = pickle.load(open('model.penguins.sav','rb'))
island_encoder = pickle.load(open('encoder.island.sav','rb'))
sex_encoder = pickle.load(open('encoder.sex.sav','rb'))
species_encoder = pickle.load(open('encoder.species.sav','rb'))
evaluations = pickle.load(open('evals.all.sav','rb'))

with tab1:
  st.header("Penquin Prediction")
  x1 = st.radio("Select island ",island_encoder.classes_)
  x1 = island_encoder.transform([x1])[0]

  x2 = st.slider("Select culmen length (mm)", 20,70,35 )
  x3 = st.slider("Select culmen depth (mm)", 10,30,15 )
  x4 = st.slider("Select flipper length (mm)", 150,250,200)
  x5 = st.slider("Select body mass (g)", 2500,6500,3000)
  x6 = st.radio("Select sex ",sex_encoder.classes_)
  x6 = sex_encoder.transform([x6])[0]

  x_new = pd.DataFrame(data=np.array([x1, x2, x3, x4, x5, x6]).reshape(1,-1), 
                 columns=['island', 'culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g', 'sex'])
  pred = model.predict(x_new)

  st.write('Predictd Species: ', species_encoder.inverse_transform(pred)[0])
  
with tab2:
  st.header("Evaluation")
  
  x = evaluations.columns
  fig = px.Figure(data=[
        px.Bar(name = 'Decision Tree',
               x = x,
               y = evaluations.loc['Decision Tress']),
        px.Bar(name = 'Random Forest',
               x = x,
               y =  evaluations.loc['Random Forest']),
        px.Bar(name = 'KNN',
               x = x,
               y =  evaluations.loc['KNN']),
        px.Bar(name = 'AdaBoost',
               x = x,
               y =  evaluations.loc['AdaBoost']),
        px.Bar(name = 'XGBoost',
               x = x,
               y =  evaluations.loc['XGBoost'])
  ])
  st.plotly_chart(fig, use_container_width=True)
  st.dataframe(evaluations)
  
with tab3:
  st.header("About")

    
