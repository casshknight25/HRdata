import pandas as pd
import plotly.express as px
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
leftdf = pd.read_csv('HRleavers.csv')


analysis = st.sidebar.selectbox("Select to view the deep dive into leavers data or EDA for predicting leavers", ("Leavers Deep Dive", "Predicting Leavers EDA"))


if analysis =='Predicting Leavers EDA':
    st.write("We can see from the correlation matrix that there is little/no correlation between features and target (i.e. that the employee left). The columns 'StandardHours' and 'Over18' are constant and hence can be dropped and they will not contribute to any model. There are null values in the columns 'NumCompaniesWorked', 'ComplaintResolved' and 'ComplaintYears' columns - this will need to be imputed before building a predicitive model. The NumCompaniesWorked values can be imputed using the difference between 'TotalWorkingYears' and 'YearsAtCompany' as these values differ by a factor of 1 year for all rows with a value of 0 for 'NumCompaniesWorked' - it has been inferred that these individuals worked for another company for a year prior to joining the current company hence giving a value of 2 for NumCompanies. The missing values in the ComplaintResolved column can be inferred from the complaintfiled column, these values are missing as these employees did not file complaints hence these will be replaced with a value of -1. The ComplaintYears field is missing values where the employee has complained and it has yet to be resolved, the amount of time is difficult to impute as taking an average of the time it took to resolve complaints will produce invalid values for employeees who complained in under a year of joining the company so it has been imputed using the YearsAtCompany field. It is worth noting that the data is moderately imbalanced, as only 16.4% of employees left, whilst this implies the company employee rentention rate is generally good it makes creating a predictive model harder as there is less data relating to the class we are trying to predict (i.e. employees who are likely to leave).")    
    HTML = 'HR_Data.html'
    with open(HTML,'r') as f: 
        html_data = f.read()
    st.components.v1.html(html_data,height=17000)

if analysis =='Leavers Deep Dive':
    leavers = alt.Chart(df).mark_bar().encode(alt.X("Left"),y="count()")
    st.altair_chart(leavers, use_container_width=True)
    st.write("We can see we have an imbalanced data set as 84% of employees have stayed with the company")
    st.header("Leavers by Department")
    lbc = alt.Chart(df).mark_bar().encode(alt.X("Department"),y="count()", color="Left")
    st.altair_chart(lbc, use_container_width=True)
    st.write("We can see that the majority of employees work in R&D, with 66% of employees in this department. HR accounts for 4% of employees and 30% work in sales. 14% of R&D employees left, 21% of Sales employees have left and 20% of HR have left. 56% of leavers were from R&D, 39% were from sales and 5% were from HR") 
    fig1 = px.box(leftdf,x='Department', y="Age")
    st.plotly_chart(fig1, use_container_width=True)
    st.write("We can see the majority of leavers between 28-39 across all departments. It is worth noting the outliers at in the age bracket 55-59, it is plausible these employees retired especially as several of these individuals had been working for 30-40 years and hence they will not be the focus of improving employee retention as retirement is less likely driven by company practises")
    fig2 = px.box(leftdf,x = 'Department', y="YearsSinceLastPromotion")
    st.plotly_chart(fig2, use_container_width=True)
    

                 
