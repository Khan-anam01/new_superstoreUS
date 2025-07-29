import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv("../datasets/SuperStoreUS.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    return df

df = load_data()

# Extract filter options
years = sorted(df['Order Date'].dt.year.unique())
regions = sorted(df['Region'].dropna().unique())
segments = sorted(df['Customer Segment'].dropna().unique())

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Analysis", "Report", "Forecast"])

# Sidebar filters
selected_year = st.sidebar.selectbox("Select Year", options=years)
selected_region = st.sidebar.selectbox("Select Region", options=["All"] + regions)
selected_segment = st.sidebar.selectbox("Select Customer Segment", options=["All"] + segments)

# Apply filters
filtered_df = df[df['Order Date'].dt.year == selected_year]
if selected_region != "All":
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]
if selected_segment != "All":
    filtered_df = filtered_df[filtered_df['Customer Segment'] == selected_segment]

# Page 1: Overview
if page == "Overview":
    st.title("📊 Business Overview")
    st.markdown("Key metrics derived from SuperStore data.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
    col2.metric("Total Profit", f"${filtered_df['Profit'].sum():,.0f}")
    col3.metric("Total Orders", filtered_df['Order ID'].nunique())

    st.subheader("Sales by Month")
    sales_monthly = filtered_df.groupby(filtered_df['Order Date'].dt.month)['Sales'].sum()
    st.bar_chart(sales_monthly)

# Page 2: Analysis
elif page == "Analysis":
    st.title("📈 Sales Performance Analysis")

    sales_by_region = filtered_df.groupby('Region')['Sales'].sum().sort_values()
    st.subheader("Sales by Region")
    st.bar_chart(sales_by_region)

    st.subheader("Top 10 Products")
    top_products = filtered_df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

    st.subheader("Customer Segment Profitability")
    seg_profit = filtered_df.groupby('Customer Segment')['Profit'].sum()
    st.bar_chart(seg_profit)

# Page 3: Report
elif page == "Report":
    st.title("📁 Customer Segmentation Report")

    clv = filtered_df.groupby(['Customer ID', 'Customer Name'])['Profit'].sum().sort_values(ascending=False).head(10)
    st.subheader("Top 10 Customers by CLV")
    st.dataframe(clv.reset_index())

    st.subheader("Customer Segment Summary")
    summary = filtered_df.groupby('Customer Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique'
    }).rename(columns={'Order ID': 'Orders'})
    st.dataframe(summary.round(2))

# Page 4: Forecast
elif page == "Forecast":
    st.title("🔮 Sales Forecast (3-Month Moving Average)")

    ts = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum().to_timestamp()
    ts_rolling = ts.rolling(window=3).mean()

    st.line_chart(ts_rolling)
    st.markdown("This 3-month moving average provides a simple forecast trend based on past sales.")
