import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('penguins.csv')
st.title('Penguins Classifier')
st.write('This app uses 6 features to identify different species of penguins using a model built on Palmer Penguins dataset. Fill the form below to get started')

rf_pickle = open('rfc_penguin.pickle','rb')
map_pickle = open('output_penguin.pickle','rb')
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])

sex = st.selectbox('Sex', options=['Female', 'Male'])

bill_length = st.number_input('Bill Length (mm)', min_value=0)

bill_depth = st.number_input('Bill Depth (mm)', min_value=0)

flipper_length = st.number_input('Flipper Length (mm)', min_value=0)

body_mass = st.number_input('Body Mass (g)', min_value=0)


Biscoe,Dream,Torgerson = 0,0,0
if island == 'Biscoe':
    Biscoe = 1
    
elif island == 'Dream':
    Dream = 1
    
elif island == 'Torgerson':
    Torgerson = 1
    
    
Male,Female =0,0
if sex == 'Male':
    Male = 1
    
elif sex == 'Female':
    Female = 1

st.write('the user inputs are {}'.format([island, sex, bill_length,bill_depth, flipper_length, body_mass]))

new_predictions = rfc.predict([[bill_length, bill_depth,flipper_length,body_mass,Biscoe,Dream,Torgerson,Female,Male]])
pred_species = unique_penguin_mapping[new_predictions]

st.write("We predict the species of your penguin is {}".format(pred_species))

st.write('We used a machine learning (Random Forest) model to predict the species, the features used in this prediction are ranked by relative importance below.')
st.image('feature_importance.png')


st.write('Below are the histograms for each continuous variable seperated by the penguins species. The vertical line represents the inputted value.')

fig, ax = plt.subplots()
ax = sns.displot(x = df['bill_length_mm'], hue = df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by species')
st.pyplot(ax)
fig,ax = plt.subplots()
ax = sns.displot(x = df['flipper_length_mm'], hue = df['species'])
plt.axvline(flipper_length)
plt.title("Flipper Length by species")
st.pyplot(ax)