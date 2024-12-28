import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import joblib
import io

# Load the Dataset
def load_data():
    data = pd.read_csv("dataset_clean.csv")
    return data

# Load the Model (if we want to use pre-trained model)
def load_model():
    model = joblib.load("house_price_model_polynomial_regression.pkl")
    return model

# Define the Streamlit App
def main():
    st.title("House Price Prediction App")
    st.write("Predict house prices based on selected features and model type.")

    # Load the dataset
    df = load_data()

    # Display the dataframe
    st.subheader("Dataset Housing Prices")
    st.write(df.head(10))

    # Feature selection using checkboxes or multiselect
    st.subheader("Select Features for Prediction")
    features = df.columns.tolist()
    selected_features = st.multiselect("Choose Features". features, default=features)

    # Prepare data for training
    X = df[selected_features]
    y = df["MEDV"]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model selection
    model_choice = st.selectbox("Select Model", ["Linear Regression", "Polynomial Regression", "Random Forest", "Pre-Trained"])

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Polynomial Regression":
        degree = st.slider("Select Degree of Polynomial", 2, 5, 2)
        poly_features = PolynomialFeatures(degree=degree)
        X_train_scaled = poly_features.fit_transform(X_train_scaled)
        X_test_scaled = poly_features.transform(X_test_scaled)
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_choice == "Pre-Trained":
        model = load_model()

    # Train the model when button is pressed
    train_button = st.button("Train Model")

    if train_button:
        # Train the model
        model.fit(X_train_scaled, y_train)

        # Predict on test data
        y_pred = model.predict(X_test_scaled)

        # Display Results
        st.subheader("Model Evaluation")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R-squared: {r2}")

        # Display Predicted vs Actual Plot
        st.subheader("Predicted vs Actual Prices")
        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual Prices", "y": "Predicted Prices"})
        st.plotly_chart(fig)

        # Option to save model
        save_model = st.button("Save Model")
        if save_model:
            model_filename = 'house_price_model.pkl'
            joblib.dump(model, model_filename)
            st.success("Model saved successfully!")

            # Create a download button for the model file
            with open(model_filename, "rb") as f:
                st.download_button(
                    label="Download Saved Model",
                    data=f,
                    file_name=model_filename,
                    mime="application/octet-stream"
                )

if __name__ == "__main__":
    main()