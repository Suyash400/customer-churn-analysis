import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



st.set_page_config(page_title="Customer Churn Prediction", layout="centered")


st.markdown("""
    <style>
    
    /* Main App Background */
    .stApp {
        background-color: #0f172a;
    }

    /* Center Title */
    h1 {
        text-align: center;
        color: white;
        font-size: 42px;
        font-weight: 700;
    }

    /* Paragraph Text */
    p {
        text-align: center;
        color: #cbd5e1;
        font-size: 18px;
    }

    /* Sidebar Color */
    section[data-testid="stSidebar"] {
        background-color: #020617;
    }

    </style>
""", unsafe_allow_html=True)
st.title("🏦 Customer Churn Prediction App")
st.write("Interactive Data Analytics project demonstrating multiple ML models.")

st.sidebar.success("Customer Churn Analytics Project")



@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)        # folder where app.py exists
    data_path = os.path.join(base_dir, "..", "data", "Churn_Modelling.csv")
    return pd.read_csv(data_path)

df = load_data()




# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# One-hot encoding Geography
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)



features = [
    'CreditScore','Gender','Age','Tenure','Balance',
    'NumOfProducts','HasCrCard','IsActiveMember',
    'EstimatedSalary','Geography_Germany','Geography_Spain'
]

X = df[features]
y = df['Exited']




scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




@st.cache_resource
def train_models(X, y):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    for m in models.values():
        m.fit(X, y)

    return models


models = train_models(X_scaled, y)




st.header("Enter Customer Details")

credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.slider("Age", 18, 90, 35)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)

num_products = st.selectbox("Number of Products", [1,2,3,4])
has_card = st.selectbox("Has Credit Card", [0,1])
active_member = st.selectbox("Is Active Member", [0,1])

salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

gender_input = st.selectbox("Gender", ["Male","Female"])
geography = st.selectbox("Geography", ["France","Germany","Spain"])



selected_model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[selected_model_name]




gender = 1 if gender_input == "Male" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0




input_data = pd.DataFrame([[
    credit_score, gender, age, tenure, balance,
    num_products, has_card, active_member,
    salary, geo_germany, geo_spain
]], columns=features)

input_scaled = scaler.transform(input_data)




if st.button("🔍 Predict Churn"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader(f"📊 Prediction using {selected_model_name}")

    if prediction == 1:
        st.error(
            f"🚨 High Churn Risk!\n\n"
            f"💔 Customer likely to leave\n\n"
            f"📉 Probability: {probability:.2f}"
        )
    else:
        st.success(
            f"🎉 Low Churn Risk!\n\n"
            f"💙 Customer likely to stay\n\n"
            f"📈 Probability: {probability:.2f}"
        )