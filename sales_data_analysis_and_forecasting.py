#step1: Import Libraries
import numpy as np              
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns


#step2: Load the Data
# Load the CSV file
df = pd.read_csv("terra1.csv",parse_dates = ['Date'])
print(df)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['Year'] = df['Date'].dt.year
#inspect the data
print(df.head())
print(df.info())

#step3:Clean and prepare the Data
#check for missing values
print(df.isnull().sum())
#drop rows with missing values
df = df.dropna()
#ensure 'Date' is date time
df['Date'] = pd.to_datetime(df['Date'])

#step4:Aggreate Data by Month
#set Date as index
df.set_index('Date', inplace = True)
#resample  monthly and sum up relevant columns
monthly = df.resample('M').sum()
print(monthly.head())

#Step5: visualize trends
import matplotlib.pyplot as plt
import seaborn as sns
#plot revenue, expenses and profit
plt.figure(figsize = (12,6))
monthly[['Total Revenue','Total Expenses','Profit']].plot()
plt.title('Monthly Revenue, Expenses and Profit')
plt.ylabel('Amount')
plt.show()

#step6:Profitability and Cost Analysis
#calculate profit margin
monthly['profit_margin'] = monthly['Profit'] / monthly['Total Revenue']
#plot profit margin
monthly['profit_margin'].plot()
plt.title('Monthly Profit Margin')
plt.ylabel('Profit Margin')
plt.show()

#step7:Forecasting with Propet
# Import Prophet for forecasting
from prophet import Prophet
#prophet expects columns 'ds' (date) and 'y' (value to forecast)
prophet_df = monthly.reset_index()[['Date', 'Profit']].rename(columns={'Date': 'ds', 'Profit': 'y'})

#Initialize and fit the model
model = Prophet()
model.fit(prophet_df)

#make a future dataframe for 24 months a head
future = model.make_future_dataframe(periods=48, freq='M')
forecast = model.predict(future)

#plot the forecast
model.plot(forecast)
plt.title('Profit Forecast')
plt.show()

# Save forecast to CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('profit_forecast.csv', index=False)

# Forecast Total Revenue
prophet_rev = monthly.reset_index()[['Date', 'Total Revenue']].rename(columns={'Date': 'ds', 'Total Revenue': 'y'})
model_rev = Prophet()
model_rev.fit(prophet_rev)
future_rev = model_rev.make_future_dataframe(periods=48, freq='M')
forecast_rev = model_rev.predict(future_rev)
model_rev.plot(forecast_rev)
plt.title('Revenue Forecast')
plt.show()
# Save forecast to CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('revenue_forecast.csv', index=False)

# Forecast Total Expenses
prophet_exp = monthly.reset_index()[['Date', 'Total Expenses']].rename(columns={'Date': 'ds', 'Total Expenses': 'y'})
model_exp = Prophet()
model_exp.fit(prophet_exp)
future_exp = model_exp.make_future_dataframe(periods=48, freq='M')
forecast_exp = model_exp.predict(future_exp)
model_exp.plot(forecast_exp)
plt.title('Expenses Forecast')
plt.show()

# Save forecast to CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('expenses_forecast.csv', index=False)

#step8:Cost breakdown Analysis
expense_cols = ['Labor Costs', 'Fertilizer Costs', 'Transport Costs']  # Update with your real column names

if all(col in monthly.columns for col in expense_cols):
    monthly[expense_cols].plot(kind='bar', stacked=True, figsize=(12,6))
    plt.title('Monthly Cost Breakdown')
    plt.ylabel('Amount')
    plt.show()
else:
    print("Update 'expense_cols' with your actual expense column names.")

#Step9:Yield Analysis $ Forecasting
#Cocoa Yield

if 'Cocoa Yield (kg)' in monthly.columns:
    prophet_yield = monthly.reset_index()[['Date', 'Cocoa Yield (kg)']].rename(columns={'Date': 'ds', 'Cocoa Yield (kg)': 'y'})
    model_yield = Prophet()
    model_yield.fit(prophet_yield)
    future_yield = model_yield.make_future_dataframe(periods=48, freq='M')
    forecast_yield = model_yield.predict(future_yield)
    model_yield.plot(forecast_yield)
    plt.title('Cocoa Yield Forecast')
    plt.show()
else:
    print("Update 'Cocoa Yield' with your actual column names.")

# Save forecast to CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('cocoa_yield_forecast.csv', index=False)

# Plantain Yield
if 'Plantain Yield (kg)' in monthly.columns:
    prophet_plantain = monthly.reset_index()[['Date', 'Plantain Yield (kg)']].rename(columns={'Date': 'ds', 'Plantain Yield (kg)': 'y'})
    model_plantain = Prophet()
    model_plantain.fit(prophet_plantain)
    future_plantain = model_plantain.make_future_dataframe(periods=48, freq='M')
    forecast_plantain = model_plantain.predict(future_plantain)
    model_plantain.plot(forecast_plantain)
    plt.title('Plantain Yield Forecast')
    plt.show()
else:
    print("Update 'Plantain Yield' with your actual  column names.")

 # Save forecast to CSV
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('plantain_yield_forecast.csv', index=False)


#step10: Export monthly summary to CSV
monthly.to_csv('monthly_summary.csv')

# Export forecasts to CSV (already included for profit, revenue, expenses)
# If not already done for yields, add:
# forecast_yield[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('cocoa_yield_forecast.csv', index=False)
# forecast_plantain[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('plantain_yield_forecast.csv', index=False)

#step11: Summerize key insights

# Print top cost drivers
print("Top cost drivers (average per month):")
print(monthly[expense_cols].mean().sort_values(ascending=False))

# Print overall profit margin
overall_profit_margin = monthly['Profit'].sum() / monthly['Total Revenue'].sum()
print(f"Overall Profit Margin: {overall_profit_margin:.2%}")



#step12: Automated summary report as text

# Gather key stats
# Gather key stats
top_cost_drivers = monthly[expense_cols].mean().sort_values(ascending=False)
overall_profit_margin = monthly['Profit'].sum() / monthly['Total Revenue'].sum()

with open('financial_summary.txt', 'w') as f:
    f.write("Terra Balance Financial Analysis Summary\n")
    f.write("="*45 + "\n\n")
    f.write("1. Key Trends:\n")
    f.write("- Revenue, expenses, and profit trends visualized monthly.\n")
    f.write("- Profit margin calculated and visualized.\n\n")
    f.write("2. Forecasts (2024–2028):\n")
    f.write("- Profit, revenue, and expenses forecasted using Prophet.\n")
    f.write("- Yield forecasts for cocoa and plantain (if available).\n\n")
    f.write("3. Cost Drivers (Average per Month):\n")
    for name, value in top_cost_drivers.items():
        f.write(f"   {name}: {value:,.2f}\n")
    f.write("\n")
    f.write(f"4. Overall Profit Margin: {overall_profit_margin:.2%}\n\n")
    f.write("5. Recommendations:\n")
    f.write("- Focus on optimizing top cost drivers.\n")
    f.write("- Use forecasts for planning and budgeting.\n")
    f.write("- Regularly update data and review forecasts.\n\n")
    f.write("6. Data Files for Further Analysis:\n")
    f.write("- monthly_summary.csv\n")
    f.write("- profit_forecast.csv\n")
    f.write("- revenue_forecast.csv\n")
    f.write("- expenses_forecast.csv\n")
    f.write("- cocoa_yield_forecast.csv (if available)\n")
    f.write("- plantain_yield_forecast.csv (if available)\n")

print("Automated summary report saved as financial_summary.txt")

#step13: Save the cleaned and processed data
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