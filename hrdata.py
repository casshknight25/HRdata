import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


st.title("HR Data Analysis")


df = pd.read_csv('HR_data.csv')
leftdf = pd.read_csv('HRLeavers.csv')


analysis = st.sidebar.selectbox("Select to view the deep dive into leavers data or EDA for predicting leavers", ("Leavers Deep Dive", "Predicting Leavers EDA"))


if analysis =='Predicting Leavers EDA':
    HTML = 'HR_data.html'
    with open(HTML,'r') as f: 
        html_data = f.read()
    st.components.v1.html(html_data,height=17000)

if analysis =='Leavers Deep Dive':
            
    fig1 = px.box(leftdf,x='Department', y="Age")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(leftdf,x = 'Department', y="YearsSinceLastPromotion")
    st.plotly_chart(fig1, use_container_width=True)


                 
