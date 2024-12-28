# House Price Prediction App

This repository contains a web-based application built using **Streamlit** to predict house prices based on user-defined input features and machine learning models. The app allows users to interactively select features, train a model, and evaluate its performance. It also enables users to download the trained model.

The dataset for this project originates from the UCI Machine Learning Repository. The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts.

Data Reference Link: https://www.kaggle.com/datasets/altavish/boston-housing-dataset

---

## Features
- **Interactive Feature Selection**: Choose specific features to include in the model.
- **Customizable Model Training**: Train models using Linear Regression, Polynomial Regression, or other techniques.
- **Real-Time Predictions**: Generate predictions for the target variable (`MEDV` - median house value).
- **Performance Metrics**: Evaluate the model using metrics like Mean Squared Error (MSE) and R-squared.
- **Download Model**: Save and download the trained model for future use.

---

## How It Works
### 1. **Dataset**
The application uses the **Boston Housing Dataset**, which includes the following features:
- `RM`: Average number of rooms per dwelling.
- `LSTAT`: Percentage of lower status of the population.
- `PTRATIO`: Pupil-teacher ratio by town.
- `MEDV`: Median value of owner-occupied homes in $1000's (target variable).

Users can upload their dataset or work with the preloaded dataset.

### 2. **Data Preprocessing**
- Handle missing values by replacing them with the mean of the respective columns.
- Detect and remove outliers using boxplots.
- Normalize features using **StandardScaler** to ensure all input features are on the same scale.

### 3. **Model Training**
Users can:
- Select features to include in the model.
- Choose a regression algorithm:
  - Linear Regression
  - Polynomial Regression
  - Additional models like Random Forest or Gradient Boosting (future updates).
- Train the model on the training dataset and visualize the results.

### 4. **Evaluation**
The application evaluates the trained model using:
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **R-squared (R²)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

### 5. **Visualization**
The app provides:
- Boxplots for visualizing outliers in features.
- Scatter plots for comparing actual vs. predicted values.

### 6. **Model Download**
Users can download the trained model as a `.joblib` file for reuse.

---

## Usage
### Prerequisites
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-price-prediction-app.git
   cd house-price-prediction-app
2. Install dependencies:
  pip install -r requirements.txt
3. Running the App
   streamlit run app.py (Open the app in your browser using)

### File Structure
- app.py: Main Streamlit application code.
- requirements.txt: Python dependencies for the project.
- HousingData.csv (optional): Dataset for prediction.
- README.md: Project documentation.

### Contributing

Feel free to fork this repository and contribute via pull requests. Any enhancements or bug fixes are welcome!

### License

The dataset for this project originates from the UCI Machine Learning Repository. The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts.

Data Reference Link: https://www.kaggle.com/datasets/altavish/boston-housing-dataset

### Author

**Farhan Wily**

