%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(page_title="Price Optimization App", layout="wide")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("/content/products_dataset 10431.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error(" Dataset not found. Please upload the correct CSV file.")
    st.stop()

# ----------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------
st.title("💹 Price Optimization based on Price Elasticity of Demand")

st.write("""
This app demonstrates a machine learning-based **price optimization model** using **XGBoost**.
It helps estimate demand and revenue changes when adjusting product prices.
""")

with st.expander("📂 View Sample Dataset"):
    st.write(df.head(10))

# Encode categorical variables
df['Seasonality'] = df['Seasonality'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['Promotion'] = df['Promotion'].map({'No': 0, 'Yes': 1})

# Remove missing values
df = df.dropna(subset=['Historical_Sales', 'Competitor_Price', 'Seasonality', 'Promotion', 'Price', 'Demand'])

# Feature Engineering
df['Price_Promo_Interaction'] = df['Price'] * df['Promotion']

# Define features and target
X = df[['Historical_Sales', 'Competitor_Price', 'Seasonality', 'Promotion', 'Price', 'Price_Promo_Interaction']]
y = df['Demand']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        reg_lambda=1.0
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# ----------------------------------------------------------
# MODEL EVALUATION
# ----------------------------------------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with st.expander("📊 Model Performance"):
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R-squared (R²):** {r2:.2f}")

# ----------------------------------------------------------
# PRICE OPTIMIZATION SECTION
# ----------------------------------------------------------
st.header("🎯 Price Optimization Tool")

# User Inputs
col1, col2 = st.columns(2)
with col1:
    product_name = st.selectbox("Select Product", df['Product_Name'].unique())
    competitor_price = st.number_input("Enter Competitor Price", min_value=0.1, max_value=100.0, value=10.0)
    seasonality = st.selectbox("Select Seasonality", ['Low', 'Medium', 'High'])
with col2:
    promotion = st.selectbox("Is Promotion Active?", ['No', 'Yes'])
    new_price = st.slider("Set New Price", min_value=0.5, max_value=100.0, value=10.0)

# Map user inputs
seasonality_mapped = {'Low': 0, 'Medium': 1, 'High': 2}[seasonality]
promotion_mapped = 1 if promotion == 'Yes' else 0

# Historical sales for selected product
historical_sales = df[df['Product_Name'] == product_name]['Historical_Sales'].mean()
price_promo_interaction = new_price * promotion_mapped

# Prepare input data
new_data = np.array([[historical_sales, competitor_price, seasonality_mapped, promotion_mapped, new_price, price_promo_interaction]])

# Predict demand and revenue
predicted_demand = model.predict(new_data)[0]
predicted_revenue = new_price * predicted_demand

st.write(f"**Predicted Demand:** {predicted_demand:.2f}")
st.write(f"**Predicted Revenue:** {predicted_revenue:.2f}")

# ----------------------------------------------------------
# REVENUE CURVE VISUALIZATION
# ----------------------------------------------------------
prices = np.linspace(0.5, 100, 60)
demands = [model.predict(np.array([[historical_sales, competitor_price, seasonality_mapped, promotion_mapped, p, p*promotion_mapped]]))[0] for p in prices]
revenues = prices * np.array(demands)

optimal_price = prices[np.argmax(revenues)]
optimal_revenue = max(revenues)

# Plot revenue curve
fig, ax = plt.subplots()
ax.plot(prices, revenues, label="Revenue Curve", color='green')
ax.axvline(optimal_price, color='blue', linestyle='--', label=f'Optimal Price: {optimal_price:.2f}')
ax.axvline(new_price, color='red', linestyle='--', label=f'Selected Price: {new_price:.2f}')
ax.set_xlabel("Price")
ax.set_ylabel("Predicted Revenue")
ax.legend()
st.pyplot(fig)

# ----------------------------------------------------------
# SUGGESTION
# ----------------------------------------------------------
st.success(f"💰 **Optimal Price Suggestion:** ₹{optimal_price:.2f}")
st.write(f"Expected Revenue at Optimal Price: ₹{optimal_revenue:.2f}")

# Compare with user's selected price
if abs(new_price - optimal_price) < 1.0:
    st.info("✅ Your selected price is near the optimal price range.")
elif predicted_revenue > (df[df['Product_Name'] == product_name]['Price'].mean() * df[df['Product_Name'] == product_name]['Demand'].mean()):
    st.success("📈 The new price is expected to increase revenue.")
else:
    st.warning("⚠️ The new price may not improve revenue compared to average.")
