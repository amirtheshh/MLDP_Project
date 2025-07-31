import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import base64

st.set_page_config(page_title="HDB Resale Price Predictor", layout="wide")


def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        img_bytes = img.read()
    b64_img = base64.b64encode(img_bytes).decode()
    page_bg = f"""
    <style>
    [data-testid="stApp"] {{
        background-image: url("data:image/png;base64,{b64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        backdrop-filter: blur(4px);
        background-color: rgba(0, 0, 0, 0.8);
        border-radius: 15px;
        padding: 2rem;
    }}
    .center-button {{
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }}
    .center-button button {{
        background-color: #00000033;
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        color: #ff4b4b;
        font-weight: bold;
        font-size: 18px;
        padding: 10px 24px;
        transition: 0.3s ease-in-out;
    }}
    .center-button button:hover {{
        background-color: #ff4b4b;
        color: white;
    }}
    .prediction-box {{
        background-color: #2E8B57;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem auto;
        text-align: center;
        max-width: 600px;
    }}
    div[data-testid="stExpander"] {{
    background-color: rgba(0, 0, 0, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
}}

div[data-testid="stExpander"] > details > summary {{
    color: white;
    font-weight: bold;
    font-size: 18px;
}}

div[data-testid="stExpander"] > details > div {{
    background-color: rgba(255, 255, 255, 0.9);
    color: black;
    padding: 1rem;
    border-radius: 0 0 12px 12px;
}}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

add_bg_from_local("hdb_bg.png")


@st.cache_resource
def load_model():
    model = joblib.load("hdb_resale_price_hgb.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    X_sample = joblib.load("X_sample.pkl")
    return model, explainer, X_sample

model, explainer, X_sample = load_model()

st.title("🏠 HDB Resale Price Prediction App")
st.markdown("### 🔍 Overview")
st.markdown("Welcome to the **HDB Resale Price Predictor**! Use the sliders and dropdowns on the left to input the flat's features, and the app will estimate its resale price using a machine learning model.")
st.markdown("### 📊 How it Works")
st.markdown("This app uses a machine learning model trained on HDB resale data to predict the resale price of a flat based on its different features. You can also view your input details and the SHAP values for a better understanding of the model's decision-making process in predicting the price.")

st.sidebar.header("🔧 Input Features")
st.sidebar.markdown("Adjust the flat's features below:")

latitude = st.sidebar.slider("Latitude", 1.27, 1.46, 1.37, 0.001, help="Latitude of the flat's location")
longitude = st.sidebar.slider("Longitude", 103.69, 103.99, 103.84, 0.001, help="Longitude of the flat's location")
closest_mrt_dist = st.sidebar.slider("Distance to Closest MRT (m)", 31.76, 3496.4, 763.94, help="Distance to the nearest MRT station in meters")
cbd_dist = st.sidebar.slider("Distance to CBD (m)", 592.12, 20225.1, 12667.79, help="Distance to the Central Business District in meters")
floor_area_sqm = st.sidebar.slider("Floor Area (sqm)", 31.0, 189.0, 94.73, help="Floor area of the flat in square meters")
years_remaining = st.sidebar.slider("Lease Years Remaining", 52, 98, 75, help="Years remaining on the flat's lease")
transaction_year = st.sidebar.selectbox("Transaction Year", [2012, 2013, 2014], help="Year of the resale transaction")

all_flat_types = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
flat_type = st.selectbox("Flat Type", all_flat_types, help="Flat Type of the HDB flat")

all_storey_ranges = ["01 TO 05", "06 TO 10", "11 TO 15", "16 TO 20", "21 TO 25", "26 TO 30", "31 TO 35", "36 TO 40"]
storey_range = st.selectbox("Storey Range", all_storey_ranges, help="The floor of the flat")

input_data = {
    'latitude': latitude,
    'longitude': longitude,
    'closest_mrt_dist': closest_mrt_dist,
    'cbd_dist': cbd_dist,
    'floor_area_sqm': floor_area_sqm,
    'years_remaining': years_remaining,
    'transaction_year': transaction_year,
}

flat_types = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
for ft in flat_types:
    input_data[f'flat_type_{ft}'] = 1 if flat_type == ft else 0

storey_ranges = ["06 TO 10", "11 TO 15", "16 TO 20", "21 TO 25", "26 TO 30", "31 TO 35", "36 TO 40"]
for sr in storey_ranges:
    input_data[f'storey_range_binned_{sr}'] = 1 if storey_range == sr else 0

input_df = pd.DataFrame([input_data])

st.markdown('<div class="center-button">', unsafe_allow_html=True)
if st.button("💰 Predict Resale Price"):
    prediction = model.predict(input_df)[0]
    
    st.markdown(
        f"""
        <div class="prediction-box">
            <h3 style="color:white;">🏡 Estimated Resale Price: ${prediction:,.2f}</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )

    with st.expander("📋 Show Input Details"):
        st.dataframe(input_df)
    
    with st.expander("📊 SHAP Feature Contribution"):
        extended_sample = pd.concat([X_sample, input_df], ignore_index=True)
        extended_shap_values = explainer(extended_sample)
        user_shap = extended_shap_values[-1]
        
        fig, ax = plt.subplots(figsize=(20, 6))
        plt.subplots_adjust(left=0.4)
        shap.plots.waterfall(user_shap, max_display=12, show=False)
        st.pyplot(fig)



        
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: lightgray;'>Created by <b>Kumar Amirtheswaran</b> | MLDP Project</p>",
    unsafe_allow_html=True
)
