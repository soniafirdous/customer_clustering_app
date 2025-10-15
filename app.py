import numpy as np
import pandas as pd
import pickle
#load the Train Kmeans model
kmeans=pickle.load(open('kmeans.pkl','rb'))

# Clustering function
def clustering(age, avg_spend, visits_per_week, promotion_interest):
    """
    Predicts the customer cluster for a new customer.

    Args:
        age: The age of the customer.
        avg_spend: The average spend of the customer.
        visits_per_week: The number of visits per week.
        promotion_interest: The promotion interest of the customer.

    Returns:
        The predicted cluster for the new customer.
    """
    # Map cluster number to names
    cluster_name = {0: "Daily", 1: "Promotion", 2: "Weekend"}
    new_customer_data = np.array([[age, avg_spend, visits_per_week, promotion_interest]])
    predicted_cluster = kmeans.predict(new_customer_data)
    return predicted_cluster[0], cluster_name[predicted_cluster[0]]

# Streamlit app
#streamlit app-------------------
import streamlit as st  
st.title("Customer clustering App")
st.write("This app helps to predict the cluster of a customer based on their features")
st.write("Please enter Customer Details:")

#row1
col1,col2=st.columns(2)

with col1:
    st.subheader("Customer Age")
    age=st.number_input("Enter Age",min_value=18,max_value=100,value=40)
with col2:
    st.subheader("Customer Spend Time")
    avg_spend=st.number_input("Enter Time",min_value=0.0,max_value=1000.0,value=30.0)
    
    
#row2
col1,col2=st.columns(2)

with col1:
    st.subheader("Visits per week")
    visits_per_week=st.number_input("Enter visits",min_value=0,max_value=20,value=4)
with col2:
    st.subheader("Promotion Interest")
    promotion_intrest=st.number_input("Promotion Interest",min_value=0,max_value=10,value=7)
    
if st.button("Predict Cluster"):
    customer_data=[[age,avg_spend,visits_per_week,promotion_intrest]]
    cluster_label=clustering(age,avg_spend,visits_per_week,promotion_intrest)
    st.success(f"The customer belongs to cluster: {cluster_label[0]}")
    