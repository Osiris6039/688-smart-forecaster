
import streamlit as st
import pandas as pd
import os
from prophet import Prophet
import base64
from datetime import datetime

DATA_FILE = "sales_data.csv"

def save_data(new_entry):
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
    df.drop_duplicates(subset=["date"], keep='last', inplace=True)
    df.to_csv(DATA_FILE, index=False)

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["date", "sales", "customers", "weather", "addons"])

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid credentials")

def forecast(df, target, periods=7):
    df = df[["date", target]].rename(columns={"date": "ds", target: "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].rename(columns={"yhat": f"{target}_forecast"})

def main_app():
    st.title("📊 AI Sales & Customer Forecast")

    with st.form("entry_form"):
        st.subheader("📥 Input Data")
        date = st.date_input("Date", value=datetime.today())
        sales = st.number_input("Sales (₱)", min_value=0)
        customers = st.number_input("Customers", min_value=0)
        weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Stormy"])
        addons = st.number_input("Add-on Sales", min_value=0)
        submitted = st.form_submit_button("Submit")

        if submitted:
            save_data({
                "date": date,
                "sales": sales,
                "customers": customers,
                "weather": weather,
                "addons": addons
            })
            st.success("Data saved!")

    df = load_data()
    if df.empty:
        st.warning("No data available yet.")
        return

    st.subheader("📂 Current Data")
    st.dataframe(df)

    st.subheader("📈 Forecast")
    forecast_sales = forecast(df, "sales")
    forecast_customers = forecast(df, "customers")
    merged = pd.merge(forecast_sales, forecast_customers, on="ds")
    forecast_result = merged.tail(7)
    st.line_chart(forecast_result.set_index("ds"))

    st.subheader("📥 Download Forecast")
    csv = forecast_result.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">📥 Download CSV</a>', unsafe_allow_html=True)

# --- Entry point ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    main_app()
else:
    login()
