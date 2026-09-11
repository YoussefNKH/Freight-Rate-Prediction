from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "catboost_freight_model.cbm"
PREPROCESSING_PATH = BASE_DIR / "models" / "catboost_preprocessing.pkl"


st.set_page_config(
	page_title="Freight Rate Predictor",
	page_icon="R",
	layout="wide",
)

st.markdown(
	"""
	<style>
	    .stApp { background: #eef3f2; color: #183437; }
	    [data-testid="stHeader"] { background: #eef3f2; }
	    [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 2rem; }
	    .brand-bar {
	        background: linear-gradient(112deg, #062f38 0%, #0b5862 68%, #0c6970 100%);
	        color: white;
	        padding: 1.8rem 2rem 1.65rem;
	        border-radius: 6px;
	        margin-bottom: 1.25rem;
	        box-shadow: 0 12px 28px rgba(6, 47, 56, 0.16);
	    }
	    .brand-kicker { color: #f08a58; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; }
	    .brand-bar h1 { margin: 0.4rem 0 0; font-size: 2.15rem; letter-spacing: -0.02em; font-weight: 750; }
	    .brand-bar p { margin: 0.55rem 0 0; color: #c9e0e1; font-size: 0.98rem; }
	    .section-label {
	        color: #0b5d68;
	        font-size: 0.7rem;
	        font-weight: 800;
	        letter-spacing: 0.15em;
	        text-transform: uppercase;
	        border-bottom: 1px solid #cbd9d8;
	        padding-bottom: 0.55rem;
	        margin: 1.1rem 0 0.8rem;
	    }
	    div[data-testid="stForm"] {
	        background: #ffffff;
	        border: 1px solid #d7e2e0;
	        border-radius: 6px;
	        padding: 0.7rem 1.25rem 1.2rem;
	        box-shadow: 0 8px 22px rgba(17, 61, 67, 0.06);
	    }
	    .rate-result {
	        background: #073b45;
	        color: white;
	        border-radius: 6px;
	        padding: 1.25rem 1.4rem;
	        margin: 1.2rem 0 0.75rem;
	        box-shadow: 0 10px 22px rgba(6, 47, 56, 0.16);
	    }
	    .rate-result small { color: #a9ced0; text-transform: uppercase; letter-spacing: 0.13em; font-size: 0.68rem; font-weight: 750; }
	    .rate-result strong { color: #ffffff; font-size: 2.45rem; line-height: 1.1; display: block; margin-top: 0.28rem; }
	    .rate-result span { color: #c9e0e1; font-size: 0.84rem; }
	    .batch-panel { background: #ffffff; border: 1px solid #d7e2e0; border-radius: 6px; padding: 1.1rem 1.25rem; }
	    button[kind="primary"] { background: #e16e3e; border-color: #e16e3e; }
	    button[kind="primary"]:hover { background: #c9572d; border-color: #c9572d; }
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
	import joblib

	model = CatBoostRegressor()
	model.load_model(str(MODEL_PATH))
	preprocessing = joblib.load(PREPROCESSING_PATH)
	return model, preprocessing


def add_date_features(data):
	data = data.copy()
	data["date"] = pd.to_datetime(data["date"], errors="coerce")
	if data["date"].isna().any():
		raise ValueError("Every date must be valid, for example 2025-12-31.")

	data["year"] = data["date"].dt.year
	data["month"] = data["date"].dt.month
	data["day"] = data["date"].dt.day
	data["day_of_week"] = data["date"].dt.dayofweek
	data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
	data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
	data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
	data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
	data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
	data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
	return data


def prepare_features(data, preprocessing):
	data = add_date_features(data)
	data["weight"] = data["weight"].fillna(preprocessing["weight_median"])
	data["market_index"] = data["market_index"].fillna(
		preprocessing["market_index_median"]
	)
	return data[preprocessing["feature_columns"]]


def predict(data, model, preprocessing):
	features = prepare_features(data, preprocessing)
	predictions = model.predict(features)
	return np.maximum(predictions, 0).round(2)


@st.cache_resource
def get_loaded_artifacts():
	if not MODEL_PATH.is_file() or not PREPROCESSING_PATH.is_file():
		return None
	return load_artifacts()


st.markdown(
	"""
	<div class="brand-bar">
	    <div class="brand-kicker">Operations / Pricing</div>
	    <h1>Freight Rate Desk</h1>
	    <p>Estimate a posted rate from lane, equipment, market, and load details.</p>
	</div>
	""",
	unsafe_allow_html=True,
)

artifacts = get_loaded_artifacts()
if artifacts is None:
	st.error(
		"Model artifacts are missing. Run the notebook deployment cells first "
		"to create models/catboost_freight_model.cbm and "
		"models/catboost_preprocessing.pkl."
	)
	st.stop()

model, preprocessing = artifacts

tab_single, tab_batch = st.tabs(["Single load", "Batch CSV"])

with tab_single:
	with st.form("single_prediction"):
		st.markdown('<div class="section-label">Lane and equipment</div>', unsafe_allow_html=True)
		lane_left, lane_right = st.columns(2)
		with lane_left:
			pickup = st.text_input("Pickup location", value="Houston")
			delivery = st.text_input("Delivery location", value="Reno")
			equipment = st.selectbox("Equipment type", ["Dry Van", "Reefer", "Flatbed"])
		with lane_right:
			distance = st.number_input("Distance (miles)", min_value=0.0, value=1687.9)
			weight = st.number_input("Weight (lb)", min_value=0.0, value=32000.0)
			date = st.date_input("Pickup date", value=pd.Timestamp("2025-12-31"))

		st.markdown('<div class="section-label">Location data</div>', unsafe_allow_html=True)
		location_left, location_right = st.columns(2)
		with location_left:
			pickup_lat = st.number_input("Pickup latitude", value=30.71157, format="%.5f")
			pickup_lon = st.number_input("Pickup longitude", value=-94.7655, format="%.5f")
		with location_right:
			delivery_lat = st.number_input("Delivery latitude", value=39.54838, format="%.5f")
			delivery_lon = st.number_input("Delivery longitude", value=-118.64843, format="%.5f")

		with st.expander("Market signals", expanded=False):
			market_index = st.number_input("Market index", value=1.0, format="%.5f")
			quote_signal = st.number_input("Quote signal", value=2.0, format="%.5f")

		submitted = st.form_submit_button("Predict rate", type="primary")

	if submitted:
		row = pd.DataFrame([{
			"pickup": pickup,
			"delivery": delivery,
			"pickup_lat": pickup_lat,
			"pickup_lon": pickup_lon,
			"delivery_lat": delivery_lat,
			"delivery_lon": delivery_lon,
			"distance": distance,
			"equipment": equipment,
			"weight": weight,
			"date": date,
			"market_index": market_index,
			"quote_signal": quote_signal,
		}])
		try:
			result = float(predict(row, model, preprocessing)[0])
			st.markdown(
				f'<div class="rate-result"><small>Estimated posted rate</small><strong>${result:,.2f}</strong><span>CatBoost freight-rate estimate</span></div>',
				unsafe_allow_html=True,
			)
			st.dataframe(
				row[["pickup", "delivery", "distance", "equipment", "weight", "date"]]
				.assign(predicted_rate=result),
				hide_index=True,
				use_container_width=True,
			)
		except (KeyError, ValueError) as error:
			st.error(str(error))

with tab_batch:
	st.markdown('<div class="batch-panel">', unsafe_allow_html=True)
	st.markdown('<div class="section-label">Batch estimation</div>', unsafe_allow_html=True)
	st.write("Run the rate desk across a validation-format CSV. Source load IDs are retained in the download for traceability.")
	uploaded_file = st.file_uploader("Upload validation-format CSV", type="csv")
	st.markdown('</div>', unsafe_allow_html=True)
	if uploaded_file is not None:
		batch = pd.read_csv(uploaded_file)
		try:
			batch["predicted_rate"] = predict(batch, model, preprocessing)
			output_columns = ["load_id", "predicted_rate"]
			output = batch[output_columns] if "load_id" in batch else batch[["predicted_rate"]]
			st.dataframe(output.head(100), use_container_width=True)
			st.download_button(
				"Download predictions",
				output.to_csv(index=False),
				"validation_predictions.csv",
				"text/csv",
			)
		except (KeyError, ValueError) as error:
			st.error(str(error))
