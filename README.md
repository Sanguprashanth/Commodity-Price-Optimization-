🧠 Price Optimization App using XGBoost
📘 Overview

This project demonstrates a machine learning-based price optimization model built with Streamlit and XGBoost.
It predicts product demand and revenue at different price points — helping identify the optimal price to maximize revenue.
The app is designed for retail and FMCG products (e.g., Milk, Wheat, Eggs, etc.) and uses price elasticity concepts to simulate pricing decisions.

🚀 Features

Interactive Streamlit web interface
Demand prediction using XGBoost Regression
Revenue optimization based on user-set price
Visual model performance metrics (MAE, R²)
Product selection, seasonality, and promotion control

🧩 Tech Stack
Category	Technologies
Frontend	Streamlit
Backend / ML	Python, Pandas, NumPy, Scikit-learn, XGBoost
Data	CSV dataset (e.g., products_dataset 10431.csv)

🧮 Model Logic

1)Encode categorical columns (Seasonality, Promotion)
2)Train XGBoost Regressor on input features
3)Predict Demand for new prices
4)Compute Revenue = Price × Predicted Demand
5)Recommend price adjustments for optimal revenue

💡 Future Improvements

->Add visual demand-vs-price curve
->Include time-series forecasting (Prophet, ARIMA)
->Integrate real-time sales data API
->Enable auto price recommendation


🧑‍💻 Author

Name:-Sangu Venkata Prashanth Reddy
Mail:-prashanthreddy.sangu@gmail.com
