import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import base64

# ✅ Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Background and Custom CSS ===
def set_bg_hack(main_bg):
    with open(main_bg, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.6);
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1);
            max-width: 1200px;
            margin: 0 auto;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background-color: rgba(255, 255, 255, 0.9) !important;
        }}
        .stButton>button {{
            background: linear-gradient(135deg, #87CEEB 0%, #1E90FF 100%) !important;
            color: black !important;
            border-radius: 8px !important;
            padding: 0.4rem 1rem !important;
            min-width: 300px !important;
            margin: 0 auto !important;
            display: block !important;
        }}
        .stButton>button:hover {{
            background: linear-gradient(135deg, #87CEEB 0%, #00BFFF 100%) !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# === Welcome Page ===
def welcome_page():
    set_bg_hack("back2.jpg")
    st.markdown("""
        <style>
        .welcome-title {
            font-size: 3rem;
            font-weight: 800;
            color: black;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .welcome-subtitle {
            font-size: 1.2rem;
            color: black;
            text-align: center;
            margin-bottom: 2.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="welcome-title">Compressive Strength Predictor</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.subheader("Dataset Selection", divider='gray')
        dataset_option = st.radio(
            "Choose a dataset:",
            [ "Upload Your Own Dataset"],
            index=0
        )

        if dataset_option == "Upload Your Own Dataset":
            uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.success("Dataset uploaded successfully!")
                    with st.expander("Preview data"):
                        st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            st.info("Using the default Concrete Compressive Strength dataset")

        if st.button("Start Analysis", type="primary"):
            st.session_state.page = "main_app"
            st.rerun()

# === Main App ===
def main_app():
    
    st.title("Concrete Compressive Strength Prediction ")

    # Custom CSS for main app
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: black;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .sub-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1E90FF;  /* Blue color */
            text-align: center;
            margin-bottom: 1rem;
        }
        .stApp {
            background-color: white;
        }
        .stButton>button {
            background: linear-gradient(135deg, #87CEEB 0%, #1E90FF 100%) !important;
            color: black !important;
            border-radius: 8px !important;
            padding: 0.4rem 1rem !important;
            min-width: 300px !important;
            margin: 0 auto !important;
            display: block !important;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #87CEEB 0%, #00BFFF 100%) !important;
        }
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.9);  /* Slightly transparent background */
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1);
            max-width: 1200px;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_data():
        df = pd.read_csv("concrete_data.csv")
        df = df.rename(columns={'fine_aggregate ': 'fine_aggregate'})
        return df

    @st.cache_resource
    def load_or_train_model():
        if os.path.exists("mlp_model.pkl"):
            model = joblib.load("mlp_model.pkl")
        else:
            df = load_data()
            X = df.drop("concrete_compressive_strength", axis=1)
            y = df["concrete_compressive_strength"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, "mlp_model.pkl")
        return model

    df = load_data()
    model = load_or_train_model()

    tab1, tab2 = st.tabs(["📊 Data Visualization", "Prediction"])

    with tab1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head())
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        st.subheader("Correlation")
        st.write(df.corr())

    with tab2:
        st.subheader("Enter Mix Details")
        col1, col2 = st.columns(2)

        with col1:
            cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=280.0)
            slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, value=70.0)
            ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0)
            water = st.number_input("Water (kg/m³)", min_value=0.0, value=180.0)

        with col2:
            superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=6.5)
            coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, value=950.0)
            fine_agg = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, value=770.0)
            age = st.number_input("Age (days)", min_value=1, value=28)

        input_data = np.array([[cement, slag, ash, water, superplasticizer, coarse_agg, fine_agg, age]])

        if st.button(" Predict Strength", type="primary"):
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Compressive Strength: {prediction:.2f} MPa")

            X = df.drop("concrete_compressive_strength", axis=1)
            y = df["concrete_compressive_strength"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = model.predict(X_test)

        

# === App state management ===
if 'page' not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    welcome_page()
else:
    main_app()
