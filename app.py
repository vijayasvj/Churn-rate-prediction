import gc

import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt


from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve, f1_score

from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgbm

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")

genai.configure(api_key=api_key)
# The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
model = genai.GenerativeModel('gemini-1.5-flash')

st.title('Telco Customer Churn Prediction')

st.markdown("""
The churn rate, also known as the rate of attrition or customer churn, is the rate at which customers stop doing 
business with an entity (e.g., Business Organization). 

You will predict the churn rate of customers in a telecom company using a stored model based on XGBoost, CatBoost or 
LightGBM.

## Instructions
1. Select the classifier (model) you want to use from the dropdown box in the sidebar.
2. To check the accuracy of the classifier, click on the **`Performance on Test Dataset`** button in the sidebar.
3. To predict the churn rate of a single observation, click on the **`Prediction on Random Instance`** button in the sidebar.
4. Or you can predict the churn rate by manual input from the sidebar, scroll down and click the **`Predict`** button.
5. The result will be displayed in the **[Prediction Result](#prediction-result)** section.

This churn rate prediction model was built by **ADRIG.AI** in collaboration with **THESTRIKE.AI** for the Gemini AI Hackathon. This Streamlit website serves as a Proof of Concept (POC) for the overall application.

## Dataset Source:
* [Kaggle Dataset URL](https://www.kaggle.com/blastchar/telco-customer-churn)
* [GitHub Dataset URL](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data)
""")


df_churn = pd.read_csv("dataset//Telco-Customer-Churn-dataset-cleaned.csv")
df_train = pd.read_csv('dataset//Telco-Customer-Churn-dataset-Train.csv', index_col=0)
df_test = pd.read_csv('dataset//Telco-Customer-Churn-dataset-Test.csv', index_col=0)

st.header('Churn Data Overview')
st.write('Data Dimension: ' + str(df_churn.shape[0]) + ' rows and ' + str(df_churn.shape[1]) + ' columns.')
st.dataframe(df_churn)


@st.cache_data
def download_dataset(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'
    return href


# st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(download_dataset(df_churn), unsafe_allow_html=True)

st.sidebar.markdown("## Predict Customer Churn Rate")
# st.sidebar.markdown("### Select a Model")
classifier_name = st.sidebar.selectbox(
    'Select a Classifier',
    ('XGBoost', 'CatBoost', 'LightGBM')
)


def get_classifier(clf_name):
    if clf_name == 'XGBoost':
        clf = xgb.XGBClassifier()  # init model
        clf.load_model("models/model_xgb.json")
    elif clf_name == 'CatBoost':
        clf = CatBoostClassifier()  # parameters not required.
        clf.load_model('models/model_catboost')
    else:
        # clf = lgbm.LGBMClassifier()
        # clf = joblib.load("models/model_lgbm.pkl")
        clf = lgbm.Booster(model_file='models/model_lgbm.txt')
    return clf


clf = get_classifier(classifier_name)


def get_transformed_data(test_data=None):
    X = df_train.drop("Churn", axis=1)

    if test_data is None:
        test_data = df_test.copy()
    # test dataset
    y_test = test_data['Churn'].values
    X_test = test_data.drop("Churn", axis=1)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = list(set(X.columns) - set(X._get_numeric_data().columns))

    ordinal_encoder = OrdinalEncoder()
    X[cat_cols] = ordinal_encoder.fit_transform(X[cat_cols])
    X_test[cat_cols] = ordinal_encoder.transform(X_test[cat_cols])

    transformer = RobustScaler()
    X[num_cols] = transformer.fit_transform(X[num_cols])
    X_test[num_cols] = transformer.transform(X_test[num_cols])

    del X
    gc.collect()
    return X_test, y_test


def make_prediction(X_test):
    try:
        # xgboost,
        test_pred = clf.predict_proba(X_test)[:, 1]  # probability of getting 1
    except AttributeError as ae:
        # lgbm load model
        # https://github.com/Microsoft/LightGBM/issues/1217
        test_pred = clf.predict(X_test)
    # st.dataframe(test_pred)
    return test_pred


if st.sidebar.button('Performance on Test Dataset'):
    X_test, y_test = get_transformed_data()
    test_pred = make_prediction(X_test)
    st.write("Performance on The Test Dataset ( ROC AUC Score) : ", roc_auc_score(y_test, test_pred))

    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = roc_curve(y_test, test_pred)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(plt)


def binning_feature(feature, value):
    bins = np.linspace(min(df_churn[feature]), max(df_churn[feature]), 4)
    if bins[0] <= value <= bins[1]:
        return 'Low'
    elif bins[1] < value <= bins[2]:
        return 'Medium'
    else:
        return 'High'

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ('Yes', 'No'))
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    phone_service = st.sidebar.selectbox('Phone Service', ('Yes', 'No', 'No phone service'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No'))
    internet_service_type = st.sidebar.selectbox('Internet Service Type', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))

    payment_method = st.sidebar.selectbox('PaymentMethod', (
        'Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))

    # tenure filter
    unique_tenure_values = df_churn.tenure.unique()
    min_value, max_value = min(unique_tenure_values), max(unique_tenure_values)

    # tenure slider
    tenure = st.sidebar.slider("Tenure", int(min_value), int(max_value), int(min_value), 1)

    # MonthlyCharges filter
    unique_monthly_charges_values = df_churn.MonthlyCharges.unique()
    min_value, max_value = min(unique_monthly_charges_values), max(unique_monthly_charges_values)

    # MonthlyCharges slider
    monthly_charges = st.sidebar.slider("Monthly Charges", min_value, max_value, float(min_value))

    min_value_total = monthly_charges * tenure
    max_value_total = (monthly_charges * tenure) + 100

    st.sidebar.markdown("**`TotalCharges`** = `MonthlyCharges` * `Tenure` + `Extra Cost ( ~100 )`")

    # TotalCharges slider
    total_charges = st.sidebar.slider("Total Charges", min_value_total, max_value_total)

    # Churn filter
    data = {'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen.lower() == 'yes' else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service_type],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'tenure-binned': binning_feature('tenure', 7),
            'MonthlyCharges-binned': binning_feature('MonthlyCharges', monthly_charges),
            'TotalCharges-binned': binning_feature('TotalCharges', total_charges)
            }

    features = pd.DataFrame(data)

    return features


input_df = user_input_features()

num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = input_df.select_dtypes(include=['object']).columns

# todo : transformer pipeline

# numerical_transformer = Pipeline(steps=[
#     ('scaler', RobustScaler())
# ])
# categorical_transformer = Pipeline(steps=[
#     ('ord', OrdinalEncoder())
# ])
# preprocessor = Pipeline(
#     steps=[
#         ('num', numerical_transformer, num_cols),
#         ('cat', categorical_transformer, cat_cols),
#     ])
# preprocessor.fit(df_churn)
#
# scaled_input = preprocessor.transform(input_df)
# st.write(scaled_input)

X = df_train.drop("Churn", axis=1)
user_df = input_df.copy()
ordinal_encoder = OrdinalEncoder()
X[cat_cols] = ordinal_encoder.fit_transform(X[cat_cols])
user_df[cat_cols] = ordinal_encoder.transform(user_df[cat_cols])

transformer = RobustScaler()
X[num_cols] = transformer.fit_transform(X[num_cols])
user_df[num_cols] = transformer.transform(user_df[num_cols])


def format_prompt(dataframe, prediction):
    user_data = dataframe.to_dict(orient='records')[0]  # Convert DataFrame to dictionary
    user_data_str = '\n'.join([f"{key}: {value}" for key, value in user_data.items()])  # Format user data as string
    prompt = (
        f"Given the following user data and the predicted churn value, provide insights on why the user is predicted to churn "
        f"and what actions can be taken to reduce the churn score and increase client success.\n\n"
        f"User Data:\n{user_data_str}\n\n"
        f"Predicted Churn: {prediction}"
    )
    return prompt

st.sidebar.markdown("##")
if st.sidebar.button('Prediction on Random Instance from Test Data'):
    st.markdown("## Prediction Result")
    random_test_instance = df_test.sample(n=1)
    X_test, y_test = get_transformed_data(random_test_instance)
    test_pred = make_prediction(X_test)
    st.markdown(f"Prediction on Random Instance From Test Data : {'**Will churn**' if test_pred[0]>0.5 else '**Will not churn**'} (Probability: {test_pred[0] : 0.2f}) ")
    st.write("Random Instance Features")
    st.dataframe(random_test_instance)
    if test_pred[0]>0.5:
        res  = "Churned"
    else:
        res = "Not churned"
    prom = format_prompt(input_df,res)
    response = model.generate_content(prom)
    st.markdown("## Gemini API Insights")
    st.markdown(response.text)








