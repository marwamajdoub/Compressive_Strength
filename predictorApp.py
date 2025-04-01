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

# Set page config
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)
# Title
st.title("🏗️ Concrete Compressive Strength Predictor")
st.markdown("""
This app predicts the compressive strength of concrete based on its composition and age.
""")
# Set background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1605106702734-205df224ecce?ixlib=rb-4.0.3");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This application uses machine learning models to predict concrete compressive strength 
based on the [Concrete Compressive Strength Dataset](https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set).
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("concrete_data.csv")
    df = df.rename(columns={'fine_aggregate ': 'fine_aggregate'})  # Fix column name
    return df

# Load or train models
@st.cache_resource
def load_models():
    # Check if saved models exist
    if all(os.path.exists(f"best_{model_name}.pkl") for model_name in ["ada", "gb", "rf"]):
        best_ada = joblib.load("best_ada.pkl")
        best_gb = joblib.load("best_gb.pkl")
        best_rf = joblib.load("best_rf.pkl")
    else:
        # Train models (this would be your actual training code)
        df = load_data()
        X = df.drop("concrete_compressive_strength", axis=1)
        y = df["concrete_compressive_strength"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Example models (replace with your actual trained models)
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
        
        # Save models
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
        st.write("**Data Types:**")
        st.write(df.dtypes)
    with col2:
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
    
    # Input widgets
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
    
    # Model selection
    model_option = st.selectbox("Select Model", 
                              ["AdaBoost", "Gradient Boosting", "Random Forest"])
    
    # Create input DataFrame
    input_data = pd.DataFrame([[cement, blast_furnace_slag, fly_ash, water, 
                              superplasticizer, coarse_aggregate, fine_aggregate, age]],
                            columns=df.columns[:-1])
    
    # Make prediction
    if st.button("Predict Strength"):
        st.subheader("Prediction Result")
        
        if model_option == "AdaBoost":
            model = best_ada
        elif model_option == "Gradient Boosting":
            model = best_gb
        else:
            model = best_rf
        
        prediction = model.predict(input_data)[0]
        
        st.metric("Predicted Compressive Strength", f"{prediction:.2f} MPa")
        
        # Show feature importance
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
    
    # Load data
    df = load_data()
    X = df.drop("concrete_compressive_strength", axis=1)
    y = df["concrete_compressive_strength"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Evaluate models
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
    
    # Display results
    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(results_df.style.format({
            "R² Score": "{:.3f}",
            "RMSE": "{:.2f}"
        }))
    
    with col2:
        fig, ax = plt.subplots()
        sns.barplot(data=results_df, x="R² Score", y="Model", ax=ax)
        ax.set_title("Model Comparison (R² Score)")
        st.pyplot(fig)
    
    # Actual vs Predicted plot
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
    
    # Water-cement ratio
    st.markdown("**Water-Cement Ratio**")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"\text{Water-Cement Ratio} = \frac{\text{Water}}{\text{Cement}}")
    with col2:
        df['water_cement_ratio'] = df['water'] / df['cement']
        st.write("Sample values:")
        st.write(df['water_cement_ratio'].head())
    
    # Aggregate ratio
    st.markdown("**Aggregate Ratio**")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"\text{Aggregate Ratio} = \frac{\text{Coarse Aggregate}}{\text{Fine Aggregate}}")
    with col2:
        df['aggregate_ratio'] = df['coarse_aggregate'] / df['fine_aggregate']
        st.write("Sample values:")
        st.write(df['aggregate_ratio'].head())
    
    # Total binder content
    st.markdown("**Total Binder Content**")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r"\text{Total Binder} = \text{Cement} + \text{Blast Furnace Slag} + \text{Fly Ash}")
    with col2:
        df['total_binder'] = df['cement'] + df['blast_furnace_slag'] + df['fly_ash']
        st.write("Sample values:")
        st.write(df['total_binder'].head())
    
    # Show correlations
    st.subheader("Correlation with Target")
    new_features = ['water_cement_ratio', 'aggregate_ratio', 'total_binder']
    correlations = df[new_features + ['concrete_compressive_strength']].corr()['concrete_compressive_strength'].drop('concrete_compressive_strength')
    
    fig, ax = plt.subplots()
    sns.barplot(x=correlations.values, y=correlations.index, ax=ax)
    ax.set_title("Correlation with Compressive Strength")
    st.pyplot(fig)