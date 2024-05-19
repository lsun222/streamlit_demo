# Run the following in terminal:
# cd Desktop/streamlit_demo
# streamlit run main.py

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Container sections
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
plotly_demo = st.container()

# Header section
with header:
    st.title("Welcome to my streamlit project")
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
    st.markdown('* **first feature:**')
    st.markdown('* **second feature:**')

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


with plotly_demo:
    st.header("Plotly Demo")
    st.text("This section demostrates the creation of animated plots with Plotly")
    df = px.data.gapminder()

    st.write(df)

    year_options=df['year'].unique().tolist()
    year = st.selectbox('Which year would you like to see?', year_options, 0)
    # df = df[df['year'] == year] # commented out for animation

    fig = px.scatter(df, x = 'gdpPercap', y = 'lifeExp',
        size = 'pop', color = 'continent', hover_name = 'continent', 
        log_x = True, size_max = 55, range_x = [100, 100000], range_y = [25, 90],
        animation_frame = 'year', animation_group = 'country')

    fig.update_layout(width = 800)
    st.write(fig)



    covid = pd.read_csv('https://raw.githubusercontent.com/shinokada/covid-19-stats/master/data/daily-new-confirmed-cases-of-covid-19-tests-per-case.csv')
    covid.columns = ['Country', 'Code', 'Date', 'Confirmed', 'Days since confirmed']
    covid['Date'] = pd.to_datetime(covid['Date']).dt.strftime('%Y-%m-%d')
    country_options = covid['Country'].unique().tolist()

    st.write(covid)

    date_options = covid['Date'].unique().tolist()
    date = st.selectbox('Which date would you like to see?', date_options, 100)
    country = st.multiselect('Which country would you like to see?', country_options, ['Brazil'])


    covid = covid[covid['Country'].isin(country)]
    # covid = covid[covid['Date'] == date] 

    fig2 = px.bar(covid, x = 'Country', y = 'Confirmed', color = 'Country',
        range_y = [0, 35000], animation_frame = 'Date', animation_group = 'Country')

    # set animation speed
    fig2.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
    fig2.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5

    fig2.update_layout(width = 800)
    st.write(fig2)
























   
