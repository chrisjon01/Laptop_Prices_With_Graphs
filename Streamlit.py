from statistics import linear_regression

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import sys

from scipy.stats import pearsonr, f_oneway
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump

# Set page configuration


# Title of the app
st.title("Laptop Price Prediction")

st.title( "Step 1: Reading the dataset")
@st.cache_data
def load_data():
    LaptopData = pd.read_csv('Laptop_price.csv', encoding="latin")
    LaptopData = LaptopData.drop_duplicates()
    return LaptopData

LaptopData = load_data()
st.write("Shape of the dataset:", LaptopData.shape)
st.write(LaptopData.head(10))


st.title("Step 2 Problem statement")
st.subheader("To create a prediction model that can predict the prices of laptops. The target variable will be the price")

st.title( "Step 3: Visualising the distribution of Target variable")
st.subheader("Distribution of Laptop Prices")
st.write("Key Observation: The target variable (price) falls into 3 distinct categories, hence why the graph appears as trimodal (3 distinct peaks)")
fig, ax = plt.subplots(figsize=(8, 3))  # Adjust the numbers as needed
LaptopData["Price"].hist(bins=50, ax=ax)
ax.set_xlabel("Price")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Laptop Prices")
st.pyplot(fig)

st.title("Step 4: Data exploration at basic level")
st.subheader("Data Information")
st.write(LaptopData.info())
st.write(LaptopData.dtypes.value_counts())

Quantative_Columns = LaptopData.select_dtypes(include=['int64', 'float64']).columns
Categorical_Columns = LaptopData.select_dtypes(include=['object']).columns
st.write("Quantitative Columns:", Quantative_Columns)
st.write("Categorical Columns:", Categorical_Columns)
st.write("Key Observation: In comparison to other data sets, Laptop data had mostly numerical values")

st.title("Step 5: Visual Exploratory Data Analysis (EDA) of data (with histogram and barcharts")
st.subheader("Quantitative Columns Distribution")
for columns in Quantative_Columns:
    fig, ax = plt.subplots()
    sns.histplot(LaptopData[columns], bins=30, kde=True, edgecolor='k', ax=ax)
    ax.set_title(f'Distribution of {columns}')
    ax.set_xlabel(columns)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

st.subheader("Categorical Columns Distribution")
for columns in Categorical_Columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(LaptopData[columns], edgecolor='k', ax=ax)
    ax.set_title(f'Distribution of {columns}')
    ax.set_xlabel(columns)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

st.title("Step 6: Removing Outliers")
st.write("Key Observation: Luckily in our data set there were not  many outliers. All the data was close to each-other")
st.subheader("Outliers in Quantitative Columns")
for col in Quantative_Columns:
    fig, ax = plt.subplots()
    sns.boxplot(LaptopData[col], ax=ax)
    ax.set_title(f'Outliers in {col}')
    st.pyplot(fig)

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

for col in Quantative_Columns:
    LaptopData = remove_outliers(LaptopData, col)

st.write('Outliers removed using IQR method')
st.write(f"Updated dataset shape: {LaptopData.shape}")

st.title("Step 7: Handling Missing Values")
st.subheader("Handling Missing Values")
st.write("Key Observation: There was not  a single missing value in the whole dataset!")

missing_values = LaptopData.isnull().sum()
st.write("Missing values in each column:\n", missing_values)

for column in LaptopData.columns:
    if LaptopData[column].isnull().any():
        if LaptopData[column].dtype == 'object':
            LaptopData[column].fillna(LaptopData[column].mode()[0], inplace=True)
        else:
            LaptopData[column].fillna(LaptopData[column].median(), inplace=True)

missing_values_after_treatment = LaptopData.isnull().sum()
st.write("\nMissing values after treatment:\n", missing_values_after_treatment)

st.title("Step 8: Feature selection")
continuous_cols = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']
categorical_cols = ['Brand']
target_col = 'Price'

# Continuous vs. Continuous (Scatter plot and Pearson correlation)
def continuous_vs_continuous(df, continuous_cols, target_col):
    correlations = {}
    for col in continuous_cols:
        fig, ax = plt.subplots()
        sns.scatterplot(x=LaptopData[col], y=LaptopData[target_col], ax=ax)
        ax.set_title(f'Scatter Plot of {col} vs {target_col}')
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        st.pyplot(fig)

        corr, _ = pearsonr(LaptopData[col], LaptopData[target_col])
        correlations[col] = corr
        st.write(f'Pearson correlation between {col} and {target_col}: {corr:.3f}')

    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    st.write("\nFeatures ranked by Pearson correlation with the target variable:")
    for feature, corr_value in sorted_correlations:
        st.write(f"{feature}: {corr_value:.3f}")

    return sorted_correlations
st.write("Key observation: From the values listed, its evident that storage capacity presents a strong correlation value")

def categorical_vs_continuous(df, categorical_cols, target_col):
    for col in categorical_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], y=LaptopData[target_col], ax=ax)
        ax.set_title(f'Box Plot of {target_col} by {col}')
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        st.pyplot(fig)

st.subheader("Continuous vs. Continuous (Scatter plot + Pearson correlation)")
sorted_corr = continuous_vs_continuous(LaptopData, continuous_cols, target_col)

st.subheader("Categorical vs. Continuous (Box plot visualization)")
categorical_vs_continuous(LaptopData, categorical_cols, target_col)

correlation_matrix = LaptopData[continuous_cols + [target_col]].corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
plt.title('Correlation Heatmap')
st.pyplot(fig)

st.title("Step 9: Statistical feature selection (categorical vs. continuous) using ANOVA test")
def categorical_vs_continuous_anova(LaptopData, categorical_cols, target_col):
    anova_results = {}
    for col in categorical_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=LaptopData[col], y=LaptopData[target_col], ax=ax)
        ax.set_title(f'Box Plot of {target_col} by {col}')
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        st.pyplot(fig)

        categories = LaptopData[col].unique()
        category_data = [LaptopData[LaptopData[col] == category][target_col] for category in categories]
        f_stat, p_value = f_oneway(*category_data)
        anova_results[col] = p_value

        st.write(f'ANOVA F-statistic for {col}: {f_stat:.3f}, p-value: {p_value:.3f}')
        if p_value < 0.05:
            st.write(f"The p-value is less than 0.05, we reject the null hypothesis. {col} has a significant effect on {target_col}.")
        else:
            st.write(f"The p-value is greater than 0.05, we fail to reject the null hypothesis. No significant effect of {col} on {target_col}.")

    return anova_results

anova_results = categorical_vs_continuous_anova(LaptopData, categorical_cols, target_col)
st.title("Step 10: Selecting final features for building AI model ")
st.subheader("Storage capacity and price have been chosen as they have strong p-values")
st.title("Step 11: Data conversion to numeric values for machine learning/predictive analysis")
LaptopData = pd.get_dummies(LaptopData, columns=['Brand'], prefix=['Brand'])
st.write(LaptopData.head())

st.title ("Step 12: Train/test data split and standardisation/normalisation of data")
X = LaptopData.drop('Price', axis=1)  # Features
y = LaptopData['Price']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f'Training features: {X_train.columns}')
st.write(f' Number of features in training data: {X_train.shape[1]}')
# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write(f'Training data shape: {X_train_scaled.shape}')
st.write(f' Testing data shape: {X_test_scaled.shape}')


st.title ("Step 13: Investigating multiple regression algorithms")
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Support Vector Regressor": SVR(),
    "AdaBoost Regressor": AdaBoostRegressor(),
    "XGBoost Regressor": XGBRegressor()
}

st.subheader("Model Training and Evaluation")
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"{model_name} results:")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"R^2: {r2:.2f}")

    # Save the model
    dump(model, f"{model_name.replace(' ', '_')}.joblib")
    dump(models["Linear Regression"], "trained_laptop_model.joblib")
    dump(scaler, 'scaler.joblib')

st.write("Models saved successfully.")

st.title ("Step 14: Selection of best model")
st.write("We have decided to use the linear regression model for final testing. It displayed a really good r^2 value and also had the least amount of error margins")

st.title("Step 15: Deployment of Linear Regression Model")

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

# Load the saved model and scaler
linear_regression = joblib.load(resource_path('trained_laptop_model.joblib'))
scaler = joblib.load(resource_path('scaler.joblib'))


# Function to predict laptop price
def predict_price(processor_speed, ram_size, storage_capacity, screen_size, weight, brand):

        # One-hot encoding for brand with **exact** feature names as used during training

        brand_map = {
            'Asus': [0, 1, 0, 0, 0],  # Assuming Asus is the second one-hot encoding column
            'Dell': [0, 0, 1, 0, 0],
            'HP': [0, 0, 0, 1, 0],
            'Lenovo': [0, 0, 0, 0, 1],
            'Acer': [1, 0, 0, 0, 0]  # Example of Acer being the first one-hot encoding column
        }
        brand_features = brand_map.get(brand, [0, 0, 0, 0, 0])  # Default to all zero if brand not found

        # Prepare input for model using **exact** feature names
        feature_names = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight',
                         'Brand_Acer', 'Brand_Asus', 'Brand_Dell', 'Brand_HP', 'Brand_Lenovo']
        input_features = np.array([[
                                       processor_speed, ram_size, storage_capacity, screen_size, weight
                                   ] + brand_features])

        # Create a DataFrame with the correct feature names
        input_df = pd.DataFrame(input_features, columns=feature_names)

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Predict the price
        predicted_price = linear_regression.predict(input_scaled)

        # Display the result
        return predicted_price[0]




# Initialize TkInter window

st.title("Laptop Price Prediction")

# Create input fields and labels

processor_speed = st.number_input("Processor Speed (GHz):", min_value=0.0)
ram_size = st.number_input("RAM Size (GB):", min_value=0.0)
storage_capacity = st.number_input("Storage Capacity (GB):", min_value=0.0)
screen_size = st.number_input("Screen Size (Inches):", min_value=0.0)
weight = st.number_input("Weight (kg):", min_value=0.0)

brand = st.selectbox("Brand:", ["Acer", "Asus", "Dell", "HP", "Lenovo"])
if st.button("Predict Price"):
    try:
        predict_price = predict_price(processor_speed, ram_size, storage_capacity, screen_size, weight, brand)
        st.success(f"Predicted Laptop Price: ${predict_price:.2f}")
    except Exception as e:
        st.error(f'An Error Has Occured ')
if __name__ == "__main__":
    st.write("Loading Model and sclaer")