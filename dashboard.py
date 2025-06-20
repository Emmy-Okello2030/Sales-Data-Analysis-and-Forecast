#step13: 
# dashboard.py
import streamlit as st
import pandas as pd

# Load your summary data
monthly = pd.read_csv('monthly_summary.csv', parse_dates=['Date'])

st.title("Terra Balance Financial Dashboard")

# Line chart for Revenue, Expenses, Profit
st.subheader("Monthly Revenue, Expenses, and Profit")
st.line_chart(monthly[['Total Revenue', 'Total Expenses', 'Profit']])

# Bar chart for Cost Breakdown
expense_cols = ['Labor Costs', 'Fertilizer Costs', 'Transport Costs']  # Update as needed
if all(col in monthly.columns for col in expense_cols):
    st.subheader("Monthly Cost Breakdown")
    st.bar_chart(monthly[expense_cols])

# Profit Margin
st.subheader("Monthly Profit Margin")
st.line_chart(monthly['profit_margin'])

# Show data table
st.subheader("Monthly Data Table")
st.dataframe(monthly)