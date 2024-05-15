# cd Desktop/streamlit_demo
# streamlit run main.py

import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Container sections
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

# Header section
with header:
    st.title("Welcome to my data science project")
    st.text("This is a test application using Streamlit.")

# Dataset section
with dataset:
    st.header("Dataset Section")
    st.text("This section can be used to display datasets.")

    taxi_data = pd.read_csv('data/train.csv')
    st.write(taxi_data.head())

    st.subheader('Distribtion of address')
    service_zone_dist = pd.DataFrame(taxi_data['ADDRESS'].value_counts()).head(50)
    st.bar_chart(service_zone_dist)

# Features section
with features:
    st.header("Features Section")
    st.text("This section can be used to discuss features of the project.")
    st.markdown('* **first feature: ** I created this feature because of this ...')
    st.markdown('* **second feature: ** I created this feature because of this ...')

# model training section
with model_training:
    st.header("Time to train the model")
    st.text("Here you get to choose the hyperparameters of the model and see how the performance change!")

    
    # create columns(horizontal sub-sections in a container)
    sel_col, disp_col = st.columns(2)

    # column 1:  create a slider: min, max. default value, step
    max_depth = sel_col.slider("What should be the max_depth of the model?", min_value = 10, max_value = 100, value = 20, step = 10)
    # create a dropdown list: options, index: default value should be the first value of the list
    n_estimators = sel_col.selectbox('How many trees should there be?', options = [100, 200, 300, 'No limit'], index = 0)
    # creat a text input feature
    input_feature = sel_col.text_input('Which feature should be used as the input feature?', 'RESALE')

    

    regr = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)
    X = taxi_data[[input_feature]]
    y = taxi_data[['TARGET']]
    regr.fit(X,y)
    prediction = regr.predict(X)

    disp_col.subheader("MSE score of model is")
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader("Mean absolute error of model is")
    disp_col.write(mean_absolute_error(y, prediction))





















   
