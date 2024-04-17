import streamlit as st
import pandas as pd
import numpy as np

st.title("Accurate")

upload_file = st.file_uploader("Upload your CSV file...", type=['csv'])
# Check if a file has been uploaded
if upload_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(upload_file)
    
    # Display the DataFrame
    st.write('**DataFrame from Uploaded CSV File:**')
    st.write(df)    

