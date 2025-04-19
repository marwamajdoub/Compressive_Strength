import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeRegressor
import joblib
import os
import base64

# Set page config
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS and background image
def set_bg_hack(main_bg):
    '''
    Set background image
    '''
    with open(main_bg, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            /* background-color: rgba(255, 255, 255, 0.4);*/  /* Near transparent */
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .main .block-container {{
                background-color: rgba(255, 255, 255, 0.6);  /* Near transparent */

            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 1px 1px rgba(0, 0, 0, 0.1);
            margin-top: 2rem;
            margin-bottom: 1rem;
            color:black;
            max-width: 900px;  /* Largeur réduite du conteneur principal */
            margin: 0 auto;    /* Centrage */
        }}
        
        [data-testid="stSidebar"] > div:first-child {{
            background-color: rgba(255, 255, 255, 0.9) !important;
        }}
        
        .stButton>button {{
           background: linear-gradient(135deg, #87CEEB 0%, #1E90FF 100%) !important;
        color: black !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.4rem 1rem !important;
        font-size: 0.9rem !important;
        transition: all 0.3s !important;
        box-shadow: 0 10px 8px rgba(0,0,0,0.1) !important;
        width: auto !important;  /* Permet au bouton de s'adapter au contenu */
        min-width: 300px !important;  /* Largeur minimale réduite */
        margin: 0 auto !important;  /* Centre le bouton */
        display: block !important;  /* Important pour le centrage */
        }}
        
        .stButton>button:hover {{
            background: linear-gradient(135deg, #87CEEB 0%, #00BFFF 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 10px 20px;
            border-radius: 5px 5px 0 0;
            transition: all 0.3s;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: #4CAF50;
            color: white;
        }}
        
        h1, h2, h3 {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Welcome Page
def welcome_page():
    set_bg_hack("back2.jpg")
    
    st.markdown("""
    <style>
        /* Force ALL text to be black */
    .main .block-container,
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container p,
    .main .block-container li,
    .main .block-container .stRadio label,
    .main .block-container .stFileUploader label,
    .main .block-container .stButton button,
    .main .block-container .stMarkdown,
    .main .block-container .stAlert,
    .main .block-container .stInfo {
        color: black !important;
    }

    /* Main title style */
    .welcome-title {
        font-size: 3rem;
        font-weight: 800;
        color: black;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Subtitle style */
    .welcome-subtitle {
        font-size: 1.2rem;
        color: black;
        text-align: center;
        margin-bottom: 2.5rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 10px;
    }
    
    .stRadio > div > label > div {
        color: white !important;
        padding: 8px 12px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: black;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        color: black;
        border: 1px dashed rgba(255,255,255,0.3);
        border-radius: 8px;
        background: rgba(255,255,255,0.05);
    }
    
    .stFileUploader > div > div:hover {
        border-color: rgba(255,255,255,0.5);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<h1 class="welcome-title"> Compressive Strength Predictor</h1>', unsafe_allow_html=True)    
    # Centered content container
    col1, col2, col3 = st.columns([1,6,1])
    with col2:
        with st.container():
            st.markdown('<div class="selection-container">', unsafe_allow_html=True)
            
            st.subheader("Dataset Selection", divider='gray')
            dataset_option = st.radio(
                "Choose a dataset:",
                ["Default  Dataset", "Upload Your Own Dataset"],
                index=0,
                help="Select the default dataset or upload your own CSV file"
            )
            
            if dataset_option == "Upload Your Own Dataset":
                uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.session_state.df = df
                        st.success("Dataset uploaded successfully!")
                        with st.expander("View data preview"):
                            st.dataframe(df.head().style.set_properties(**{
                                'background-color': 'rgba(255,255,255,0.1)',
                                'color': 'white',
                                'border-color': 'rgba(255,255,255,0.2)'
                            }))
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
            else:
                st.info("Using the default Concrete Compressive Strength dataset")
            
            if st.button("Start Analysis", type="primary"):
                st.session_state.page = "main_app"
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
# Main App
def main_app():
    set_bg_hack("back2.jpg")
    
    # Sidebar
    st.sidebar.header("Navigation")
    st.sidebar.info("""
    This application uses machine learning models to predict concrete compressive strength.
    """)
    
    if st.sidebar.button("← Back to Welcome Page"):
        st.session_state.page = "welcome"
        st.rerun()
    
    # Title
    st.title("🏗️ Concrete Compressive Strength Predictor")
    
    # Load data
    @st.cache_data
    def load_data():
        if 'df' in st.session_state:
            return st.session_state.df
        else:
            df = pd.read_csv("concrete_data.csv")
            df = df.rename(columns={'fine_aggregate ': 'fine_aggregate'})
            return df

    # Load or train models
    @st.cache_resource
    def load_models():
        if all(os.path.exists(f"best_{model_name}.pkl") for model_name in ["ada", "gb", "rf"]):
            best_ada = joblib.load("best_ada.pkl")
            best_gb = joblib.load("best_gb.pkl")
            best_rf = joblib.load("best_rf.pkl")
        else:
            df = load_data()
            X = df.drop("concrete_compressive_strength", axis=1)
            y = df["concrete_compressive_strength"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            best_ada = AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=3),
                n_estimators=150,
                learning_rate=0.1,
                random_state=42
            )
            best_ada.fit(X_train, y_train)
            
            best_gb = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            best_gb.fit(X_train, y_train)
            
            best_rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
            best_rf.fit(X_train, y_train)
            
            joblib.dump(best_ada, "best_ada.pkl")
            joblib.dump(best_gb, "best_gb.pkl")
            joblib.dump(best_rf, "best_rf.pkl")
        
        return best_ada, best_gb, best_rf

    best_ada, best_gb, best_rf = load_models()

    # App sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Exploration", "🤖 Model Prediction", "📈 Model Comparison", "⚙️ Feature Engineering"])

    with tab1:
        st.header("Data Exploration")
        df = load_data()
        
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data")
            st.write(df)
        
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.write("**Data Types:**")
                st.write(df.dtypes)
        with col2:
            with st.container(border=True):
                st.write("**Descriptive Statistics:**")
                st.write(df.describe())
        
        st.subheader("Data Visualization")
        plot_type = st.selectbox("Select plot type", 
                                ["Histogram", "Box Plot", "Correlation Heatmap", "Scatter Plot"])
        
        if plot_type == "Histogram":
            column = st.selectbox("Select column", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)
        
        elif plot_type == "Box Plot":
            column = st.selectbox("Select column", df.columns)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, y=column, ax=ax)
            st.pyplot(fig)
        
        elif plot_type == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
        
        elif plot_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", df.columns)
            with col2:
                y_axis = st.selectbox("Y-axis", df.columns)
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)

    with tab2:
        st.header("Concrete Strength Prediction")
        
        st.subheader("Input Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            cement = st.slider("Cement (kg/m³)", 
                              min_value=100.0, max_value=600.0, value=280.0, step=1.0)
            blast_furnace_slag = st.slider("Blast Furnace Slag (kg/m³)", 
                                          min_value=0.0, max_value=400.0, value=73.5, step=1.0)
            fly_ash = st.slider("Fly Ash (kg/m³)", 
                               min_value=0.0, max_value=200.0, value=0.0, step=1.0)
            water = st.slider("Water (kg/m³)", 
                             min_value=120.0, max_value=250.0, value=181.6, step=0.1)
        
        with col2:
            superplasticizer = st.slider("Superplasticizer (kg/m³)", 
                                        min_value=0.0, max_value=40.0, value=6.4, step=0.1)
            coarse_aggregate = st.slider("Coarse Aggregate (kg/m³)", 
                                        min_value=700.0, max_value=1200.0, value=968.0, step=1.0)
            fine_aggregate = st.slider("Fine Aggregate (kg/m³)", 
                                      min_value=500.0, max_value=1000.0, value=773.6, step=1.0)
            age = st.slider("Age (days)", 
                           min_value=1, max_value=365, value=28)
        
        model_option = st.selectbox("Select Model", 
                                  ["AdaBoost", "Gradient Boosting", "Random Forest"])
        
        input_data = pd.DataFrame([[cement, blast_furnace_slag, fly_ash, water, 
                                  superplasticizer, coarse_aggregate, fine_aggregate, age]],
                                columns=df.columns[:-1])
        
        if st.button("Predict Strength", type="primary"):
            st.subheader("Prediction Result")
            
            if model_option == "AdaBoost":
                model = best_ada
            elif model_option == "Gradient Boosting":
                model = best_gb
            else:
                model = best_rf
            
            prediction = model.predict(input_data)[0]
            
            if prediction < 20:
                color = "#ff4444"
            elif prediction < 40:
                color = "#ffbb33"
            else:
                color = "#00C851"
                
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: {color}; font-size: 2.5rem;">{prediction:.2f} MPa</h3>
                <p>Predicted Compressive Strength</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Feature Importance")
            
            try:
                feature_imp = pd.DataFrame({
                    'Feature': input_data.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots()
                sns.barplot(data=feature_imp, x='Importance', y='Feature', ax=ax)
                ax.set_title(f"{model_option} Feature Importance")
                st.pyplot(fig)
            except:
                st.warning("Feature importance not available for this model")

    with tab3:
        st.header("Model Comparison")
        
        df = load_data()
        X = df.drop("concrete_compressive_strength", axis=1)
        y = df["concrete_compressive_strength"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def evaluate_model(model, name):
            y_pred = model.predict(X_test)
            return {
                "Model": name,
                "R² Score": r2_score(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        results = [
            evaluate_model(best_ada, "AdaBoost"),
            evaluate_model(best_gb, "Gradient Boosting"),
            evaluate_model(best_rf, "Random Forest")
        ]
        results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False)
        
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(results_df.style.format({
                "R² Score": "{:.3f}",
                "RMSE": "{:.2f}"
            }).highlight_max(subset=["R² Score"], color='#90EE90')
                         .highlight_min(subset=["RMSE"], color='#90EE90'))
        
        with col2:
            fig, ax = plt.subplots()
            sns.barplot(data=results_df, x="R² Score", y="Model", ax=ax)
            ax.set_title("Model Comparison (R² Score)")
            st.pyplot(fig)
        
        st.subheader("Actual vs Predicted Values")
        model_for_plot = st.selectbox("Select model to plot", 
                                    ["AdaBoost", "Gradient Boosting", "Random Forest"])
        
        if model_for_plot == "AdaBoost":
            y_pred = best_ada.predict(X_test)
        elif model_for_plot == "Gradient Boosting":
            y_pred = best_gb.predict(X_test)
        else:
            y_pred = best_rf.predict(X_test)
        
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Strength")
        ax.set_ylabel("Predicted Strength")
        ax.set_title(f"{model_for_plot} - Actual vs Predicted")
        st.pyplot(fig)

    with tab4:
        st.header("Feature Engineering")
        df = load_data()
        
        st.subheader("Create New Features")
        
        with st.expander("Water-Cement Ratio"):
            col1, col2 = st.columns(2)
            with col1:
                st.latex(r"\text{Water-Cement Ratio} = \frac{\text{Water}}{\text{Cement}}")
            with col2:
                df['water_cement_ratio'] = df['water'] / df['cement']
                st.write("Sample values:")
                st.write(df['water_cement_ratio'].head())
        
        with st.expander("Aggregate Ratio"):
            col1, col2 = st.columns(2)
            with col1:
                st.latex(r"\text{Aggregate Ratio} = \frac{\text{Coarse Aggregate}}{\text{Fine Aggregate}}")
            with col2:
                df['aggregate_ratio'] = df['coarse_aggregate'] / df['fine_aggregate']
                st.write("Sample values:")
                st.write(df['aggregate_ratio'].head())
        
        with st.expander("Total Binder Content"):
            col1, col2 = st.columns(2)
            with col1:
                st.latex(r"\text{Total Binder} = \text{Cement} + \text{Blast Furnace Slag} + \text{Fly Ash}")
            with col2:
                df['total_binder'] = df['cement'] + df['blast_furnace_slag'] + df['fly_ash']
                st.write("Sample values:")
                st.write(df['total_binder'].head())
        
        st.subheader("Correlation with Target")
        new_features = ['water_cement_ratio', 'aggregate_ratio', 'total_binder']
        correlations = df[new_features + ['concrete_compressive_strength']].corr()['concrete_compressive_strength'].drop('concrete_compressive_strength')
        
        fig, ax = plt.subplots()
        sns.barplot(x=correlations.values, y=correlations.index, ax=ax)
        ax.set_title("Correlation with Compressive Strength")
        st.pyplot(fig)

# App state management
if 'page' not in st.session_state:
    st.session_state.page = "welcome"

# Page routing
if st.session_state.page == "welcome":
    welcome_page()
else:
    main_app()