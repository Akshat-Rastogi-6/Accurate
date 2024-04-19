from sklearn.calibration import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# --server.enableXsrfProtection false
def switch_case(argument):
    switcher = {
        "Random Forest Classifier": RandomForestClassifier(),
        "SVM": SVC(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Adaboost Classifier" : AdaBoostClassifier(),
        "Extra Trees Classifier" :ExtraTreeClassifier(),
    }
    return switcher.get(argument)


def main():
    st.title("Accurate")
    upload_file = st.file_uploader("Upload your CSV file...", type=['csv'])
    # Check if a file has been uploaded
    if upload_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(upload_file)
        # Display the DataFrame
        st.write('**DataFrame from Uploaded CSV File:**')
        st.write(df.head())

        preprocessing(df)

        model_name = st.sidebar.selectbox("Select Machine Learning Model :", ["Random Forest Classifier","SVM","Decision Tree Classifier","Logistic Regression", "Adaboost Classifier","Extra Trees Classifier"])
        model = switch_case(model_name)
        run_model(df, model)

def preprocessing(df):
    encoder = LabelEncoder()
    # Iterate through each column in the dataframe
    for column in df.columns:
        # Check if the column contains string values
        if df[column].dtype == 'object':
            # Fit label encoder and transform the column
            df[column] = encoder.fit_transform(df[column])

    df = df.fillna(df.mean())
    

def run_model(df, model):
    st.subheader(switch_case(model))
    
    target_column = st.text_input("Enter your target column : ")
    # Prepare data
    if target_column != "" :
        X = df.drop(columns=[target_column]) # replace 'target_column' with the name of your target column
        y = df[target_column] # replace 'target_column' with the name of your target column
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train mode
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)    

main()



