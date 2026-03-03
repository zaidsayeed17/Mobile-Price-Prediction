import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import numpy as np

# Load cleaned data
data = pd.read_csv('Mobiles_Cleaned.csv')

# Encoding categorical features
brand_encoder = LabelEncoder()
model_encoder = LabelEncoder()

data['Brand_enc'] = brand_encoder.fit_transform(data['Brand'])
data['Model_enc'] = model_encoder.fit_transform(data['Model'])

features = data[['Brand_enc', 'Model_enc', 'Ram', 'Storage']]
target = data['Selling Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title("Mobile Price Prediction")

brand_list = data['Brand'].unique().tolist()
selected_brand = st.selectbox('Select Brand', brand_list)

# Filter models of selected brand
models_for_brand = data[data['Brand'] == selected_brand]['Model'].unique().tolist()
selected_model = st.selectbox('Select Model', models_for_brand)

ram_options = sorted(data[(data['Brand'] == selected_brand) & (data['Model'] == selected_model)]['Ram'].unique())
selected_ram = st.selectbox('Select RAM (GB)', ram_options)

storage_options = sorted(data[(data['Brand'] == selected_brand) & (data['Model'] == selected_model)]['Storage'].unique())
selected_storage = st.selectbox('Select Storage (GB)', storage_options)

if st.button('Predict Price'):
    # Encode inputs
    brand_val = brand_encoder.transform([selected_brand])[0]
    model_val = model_encoder.transform([selected_model])[0]

    input_features = [[brand_val, model_val, selected_ram, selected_storage]]
    price_pred = model.predict(input_features)[0]
    
    st.markdown(
        f"""
        <div style="
            background-color: #ff7985;  /* Blue shade: aap koi bhi HEX color de sakte hain */
            padding: 24px;
            border-radius: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(30, 144, 255, 0.15);
        ">
            <span style="
                color: #e5f7a3;
                font-size: 2.2rem;   /* Large font */
                font-weight: bold;
            ">
                Estimated Selling Price: ₹{price_pred:.2f}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM2xnYzVjNndqaGhydm1xNGV3cXA4aGg1OHBxZHE3ZDRsY29wc3FxZiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/BHNfhgU63qrks/giphy.gif ");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
