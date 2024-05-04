from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder, label_binarize
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
from sklearn.metrics import accuracy_score, auc, cohen_kappa_score, confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# --server.enableXsrfProtection false

# This function helps you to choose between various different machine learning models.
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

# This function helps you generate a confusion matrix for your selected model
def cm(y_test, y_pred):
    mdl_cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (13,12))
    sns.heatmap(mdl_cm, annot=True)
    plt.savefig('uploads/confusion_matrix.jpg', format="jpg", dpi=300)
    st.image("uploads/confusion_matrix.jpg", caption="Confusion Matrix of your Data", width=600)\
    
# All the pre-processing like removing null values, label encoding is done here
def pre_processing(df, columns):
    encoder = LabelEncoder()
    # Iterate through each column in the dataframe
    for column in df.columns:
        # Check if the column contains string values
        if df[column].dtype == 'object':
            # Fit label encoder and transform the column
            df[column] = encoder.fit_transform(df[column])

    imputer = SimpleImputer(strategy='mean')
    df = imputer.fit_transform(df)

    # WILL DO THIS LATER 

    # st.write("Do you want to do Feature Engineering to make the results better.")
    # if st.button("Yes") :
    #     st.write("Choose the elements you want to do feature engineering with : ")
    #     for i in columns :
    #         st.button(i)

    # if st.button("No") :
    #     pass

    return df
    
# This function does the data splitting and applies ML
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
            
            testing_file = st.sidebar.file_uploader("Upload your testing CSV file...", type=['csv'])

            if testing_file is not None:
                test = pd.read_csv(testing_file)
                test_columns = test.columns

                test = pre_processing(test, test_columns)
                test = pd.DataFrame(df, columns=test_columns)

                y_pred = model.predict(test)

            y_pred = model.predict(X_test) #this line tests your data for code from the dataset.
            
            eval_mat = st.sidebar.selectbox("Select your Evaluation Matrix :", ["Accuracy", "Precision", "Recall(Sensitivity)", "F1 Score", "Roc AUC Score", "Cohen's Kappa", "Matthew's Correlation Coefficient", "Log Loss"])

            if eval_mat == "Accuracy" :
                # Display accuracy
                result = str(accuracy_score(y_test, y_pred))
                st.write("Accuracy :", float(result))

            if eval_mat == "Precision":
                result = str(precision_score(y_test, y_pred, average='weighted'))
                st.write("Precision:", float(result))

            if eval_mat == "Recall(Sensitivity)":
                result = str(recall_score(y_test, y_pred, average='weighted'))
                st.write("Recall(Sensitivity)c:", float(result))

            if eval_mat == "F1 Score":
                result = str(f1_score(y_test, y_pred, average='weighted'))
                st.write("F1 Score :", float(result))
            
            if eval_mat == "Roc AUC Score":
                result = str(roc_auc_score(y_test, y_pred))
                st.write("Roc AUC Score :", float(result))
                
            # if eval_mat == "Confusion Matrix":
            #     result = str(confusion_matrix(y_test, y_pred))
            #     st.write("Confusion Matrix :", float(result))

            if eval_mat == "Cohen's Kappa":
                result = str(cohen_kappa_score(y_test, y_pred))
                st.write("Cohen's Kappa :", float(result))

            if eval_mat == "Matthew's Correlation Coefficient":
                result = str(matthews_corrcoef(y_test, y_pred))
                st.write("Matthew's Correlation Coefficient :", float(result))

            if eval_mat == "Log Loss":
                result = str(log_loss(y_test, y_pred))
                st.write("Log Loss :", float(result))

            if result != "" :
                curves = st.sidebar.selectbox("Select the metrics you want to see : ", ["Confusion Matrix", "ROC Curve"])
                if curves == "Confusion Matrix":
                    cm(y_test, y_pred)
                if curves == "ROC Curve" :
                    y_arr = np.array(y)
                    unique_classes, counts = np.unique(y_arr, return_counts=True)
                    n_classes = len(unique_classes)
                    y_train_bin = label_binarize(y_train, classes=range(n_classes))
                    # Train the classifier
                    classifier = OneVsRestClassifier(model)  # Your chosen classifier
                    classifier.fit(X_train, y_train_bin)

                    # Get predicted probabilities
                    y_score = classifier.predict_proba(X_test)
                    aoc(y_score, n_classes, y_test)


def aoc(y_score, n_classes, y_test):   
    y_test_np = y_test.to_numpy() 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_np == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiclass Data')
    plt.legend(loc="lower right")
    plt.savefig('uploads/roc.jpg', format="jpg", dpi=300)
    st.image("uploads/roc.jpg", caption="ROC of your Data", width=600)

# def visualization():
#     precision, recall, _ = precision_recall_curve(y_true, y_score)
#     plt.plot(recall, precision, marker='.')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.show()



# This is the main function
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

        # visualization()

        columns = df.columns
        df = pre_processing(df, columns)

        df = pd.DataFrame(df, columns=columns)

        model_name = st.sidebar.selectbox("Select Machine Learning Model :", ["Random Forest Classifier","SVM","Decision Tree Classifier","Logistic Regression", "Adaboost Classifier","Extra Trees Classifier","Gradient Boosting Classifier","K-Nearest Neighbors Classifier", "Gaussian Naive Bayes Classifier", "Bernoulli Naive Bayes Classifier", "Multinomial Naive Bayes Classifier", "Passive Aggressive Classifier", "Ridge Classifier", "Lasso Classifier", "ElasticNet Classifier", "Bagging Classifier", "Stochastic Gradient Descent Classifier", "Perceptron", "Isolation Forest", "Principal Component Analysis (PCA)", "Linear Discriminant Analysis (LDA)", "Quadratic Discriminant Analysis (QDA)", "XGBoost Classifier", "LightGBM Classifier", "CatBoost Classifier", "MLP Classifier"])

        model = switch_case(model_name)
        run_model(df, model, model_name)

        st.sidebar.button("Print out Dataset Details.")

main()



