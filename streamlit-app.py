from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from xgboost import XGBClassifier

# --server.enableXsrfProtection false
def switch_case(argument):
    switcher = {
        "Random Forest Classifier": RandomForestClassifier(),
        "SVM": SVC(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Adaboost Classifier" : AdaBoostClassifier(),
        "Extra Trees Classifier" :ExtraTreeClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
        "Gaussian Naive Bayes Classifier": GaussianNB(),
        "Bernoulli Naive Bayes Classifier": BernoulliNB(),
        "Multinomial Naive Bayes Classifier": MultinomialNB(),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(),
        "Ridge Classifier": RidgeClassifier(),
        "Lasso Classifier": Lasso(),
        "ElasticNet Classifier": ElasticNet(),
        "Bagging Classifier": BaggingClassifier(),
        "Stochastic Gradient Descent Classifier": SGDClassifier(),
        "Perceptron": Perceptron(),
        "Isolation Forest": IsolationForest(),
        "Principal Component Analysis (PCA)": PCA(),
        "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis(),
        "XGBoost Classifier": XGBClassifier(),
        "LightGBM Classifier": LGBMClassifier(),
        "CatBoost Classifier": CatBoostClassifier(),
        "MLP Classifier": MLPClassifier()
    }
    return switcher.get(argument)


def cm(y_test, y_pred):
    mdl_cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (13,12))
    sns.heatmap(mdl_cm, annot=True)
    plt.savefig('uploads/confusion_matrix.jpg', format="jpg", dpi=300)
    st.image("uploads/confusion_matrix.jpg", caption="Confusion Matrix of your Data", width=600)

def pre_processing(df):
    encoder = LabelEncoder()
    # Iterate through each column in the dataframe
    for column in df.columns:
        # Check if the column contains string values
        if df[column].dtype == 'object':
            # Fit label encoder and transform the column
            df[column] = encoder.fit_transform(df[column])

    imputer = SimpleImputer(strategy='mean')
    df = imputer.fit_transform(df)
    

def run_model(df, model, model_name):
    st.subheader(model_name)
    
    target_column = st.text_input("Enter your target column : ")
    # Prepare data
    if target_column != "" :
        X = df.drop(columns=[target_column]) # replace 'target_column' with the name of your target column
        y = df[target_column] # replace 'target_column' with the name of your target column
        testing_size = st.text_input("Enter the test splitting size : ")
        if testing_size != "":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(testing_size), random_state=42)

            # Train mode
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Display accuracy
            accuracy = str(accuracy_score(y_test, y_pred))
            st.write("Accuracy:", float(accuracy))

            if accuracy != "" :
                curves = st.sidebar.selectbox("Select the metrics you want to see : ", ["Confusion Matrix", "ROC Curve"])
                if curves == "Confusion Matrix":
                    cm(y_test, y_pred)
                if curves == "ROC Curve" :
                    aoc(y_test, y_pred)

                
def aoc(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('uploads/roc.jpg', format="jpg", dpi=300)
    st.image("uploads/roc.jpg", caption="Confusion Matrix of your Data", width=600)

def main():
    st.title("Accurate 🎯")
    upload_file = st.file_uploader("Upload your CSV file...", type=['csv'])
    # Check if a file has been uploaded
    if upload_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(upload_file)
        # Display the DataFrame
        st.write('**DataFrame from Uploaded CSV File:**')
        st.write(df.head())

        pre_processing(df)

        model_name = st.sidebar.selectbox("Select Machine Learning Model :", ["Random Forest Classifier","SVM","Decision Tree Classifier","Logistic Regression", "Adaboost Classifier","Extra Trees Classifier","Gradient Boosting Classifier","K-Nearest Neighbors Classifier", "Gaussian Naive Bayes Classifier", "Bernoulli Naive Bayes Classifier", "Multinomial Naive Bayes Classifier", "Passive Aggressive Classifier", "Ridge Classifier", "Lasso Classifier", "ElasticNet Classifier", "Bagging Classifier", "Stochastic Gradient Descent Classifier", "Perceptron", "Isolation Forest", "Principal Component Analysis (PCA)", "Linear Discriminant Analysis (LDA)", "Quadratic Discriminant Analysis (QDA)", "XGBoost Classifier", "LightGBM Classifier", "CatBoost Classifier", "MLP Classifier"])

        model = switch_case(model_name)
        run_model(df, model, model_name)


main()



