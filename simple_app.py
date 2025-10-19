import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px


# Title
st.title('My First Streamlit App')

# Text
st.write('Hello, Streamlit!')

# Create some sample data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'z': np.random.randn(100)
})


# Display data
st.subheader('Sample Data')
st.dataframe(data)

# Create a chart
st.subheader('Line Chart')
st.line_chart(data)

# # create a map
# st.subheader('Map')
# st.map(data.rename(columns={'x': 'lat', 'y': 'lon'}))

# Text input
name = st.text_input('Enter your name')

st.subheader(name)

# Number input
age = st.number_input('Enter your age', min_value=0, max_value=120)

# Selectbox
option = st.selectbox('Choose an option', ['Option 1', 'Option 2', 'Option 3'])

# Slider
value = st.slider('Select a value', 0, 100, 50)

# Checkbox
if st.checkbox('Show details'):
    st.write('Here are the details...')

# Button
if st.button('Click me'):
    st.write('Button was clicked!')

st.set_page_config(page_title="Sales Dashboard", layout="wide")

st.title("📊 Sales Dashboard")

# Sidebar for filters
st.sidebar.header("Filters")
region = st.sidebar.selectbox("Select Region", ["North", "South", "East", "West"])
date_range = st.sidebar.date_input("Select Date Range", value=pd.to_datetime("2023-01-01"))

# Generate sample data
@st.cache_data 
def load_data():
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    data = pd.DataFrame({
        "date": dates,
        "sales": np.random.randint(1000, 5000, 365),
        "region": np.random.choice(["North", "South", "East", "West"], 365),
        "product": np.random.choice(["Product A", "Product B", "Product C"], 365)
    })
    return data

data = load_data()

# Filter data based on selection
filtered_data = data[data['region'] == region]

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sales", f"${filtered_data['sales'].sum():,}")
with col2:
    st.metric("Average Daily Sales", f"${filtered_data['sales'].mean():.0f}")
with col3:
    st.metric("Number of Days", len(filtered_data))

# Charts
fig = px.line(filtered_data, x='date', y='sales', title=f'Sales Trend for {region} Region')
st.plotly_chart(fig, use_container_width=True)

# Data table
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(filtered_data)