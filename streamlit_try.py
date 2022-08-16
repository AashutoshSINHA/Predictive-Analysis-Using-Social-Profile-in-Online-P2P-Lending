import streamlit as st
import cv2
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from prediction import get_prediction


st.set_page_config(page_title='Site Energy Intensity Prediction', page_icon="💸",
                               layout="wide", initial_sidebar_state='expanded')

pickle_in = open('model.pkl', 'rb') 
model = pickle.load(pickle_in)


# creating option list for dropdown menu

features = ['ListingNumber', 'ClosedDate', 'BorrowerAPR', 'BorrowerRate','DateCreditPulled', 'LoanMonthsSinceOrigination', 'LoanNumber',
       'LoanOriginationQuarter', 'LP_CustomerPayments','LP_CustomerPrincipalPayments']

st.markdown("<h1 style='text-align: center;'>Prosper LoanStatus Prediction App 💸 </h1>", unsafe_allow_html=True)


def main():
    with st.form('prediction_form'):

        st.header("Predict the input for following features:") 
                
        
        ListingNumber = st.selectbox( 'ListingNumber:', [1,4,6])
        
        ClosedDate = st.selectbox( 'ClosedDate:', [1,78,81,92,100,105])
        BorrowerAPR = st.selectbox('BorrowerAPR:', [1,747,902,1260,1644,3672])
        BorrowerRate = st.selectbox( 'BorrowerRate:', [1,1319,1508,1651,1905,3672])
        DateCreditPulled = st.selectbox('DateCreditPulled:', [1,4,6])
        LoanMonthsSinceOrigination = st.selectbox('LoanMonthsSinceOrigination:', [8,9,13,20,4336,4485,4899,5215,5865])
        LoanNumber = st.selectbox('LoanNumber:', [1,4,6])
        LoanOriginationQuarter = st.selectbox('LoanOriginationQuarter:', [13,585,1243,1270,1600,2403,3074,3913,4424,5061,14450])
        LP_CustomerPayments = st.selectbox('LP_CustomerPayments:', [1,82,83,119,6208])
        LP_CustomerPrincipalPayments = st.selectbox('LP_CustomerPrincipalPayments:', [1,1938,2042,2274,2595,6308])
        
        submit = st.form_submit_button("Predict")
    
    if submit:

        data= np.array([ListingNumber, ClosedDate, BorrowerAPR, BorrowerRate, DateCreditPulled,LoanMonthsSinceOrigination,LoanNumber,
                        LoanOriginationQuarter,LP_CustomerPayments,LP_CustomerPrincipalPayments]).reshape(1, -1)
        
        pred = get_prediction(data=data, model=model)

        if pred == 0:
            LoanStatus = 'Rejected'
        elif pred == 1:
            LoanStatus = 'Accepted'
            
            
        st.write(f"The predicted LoanStatus is:  {LoanStatus}")
