from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import datetime

app = Flask(__name__)

# Load the Bitcoin price dataset
df = pd.read_csv("bitcoin_price.csv")

# Preprocess the data
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

numeric_cols = ['Open', 'High', 'Low', 'Price']
df[numeric_cols] = df[numeric_cols].replace([',', '%'], ['', ''], regex=True)
df[numeric_cols] = df[numeric_cols].astype(float)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[numeric_cols])
df_scaled = pd.DataFrame(scaled_features, columns=[f'{col}_scaled' for col in numeric_cols])
df = pd.concat([df, df_scaled], axis=1)

features = ['Open_scaled', 'High_scaled', 'Low_scaled', 'Year', 'Month', 'DayOfWeek']
target = 'Price_scaled'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Exchange rate from BTC to INR
EXCHANGE_RATE = 2500000  # Update this as needed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        current_date = datetime.datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        current_day_of_week = current_date.weekday()

        today_data = pd.DataFrame({
            'Year': [current_year],
            'Month': [current_month],
            'DayOfWeek': [current_day_of_week],
            'Open_scaled': [df['Open_scaled'].iloc[-1]],
            'High_scaled': [df['High_scaled'].iloc[-1]],
            'Low_scaled': [df['Low_scaled'].iloc[-1]]
        })

        today_data = today_data[features]
        today_predicted_scaled = model.predict(today_data)[0]

        # Transform the scaled prediction back to the actual price
        price_mean = scaler.mean_[3]
        price_std = scaler.scale_[3]
        today_predicted_price = today_predicted_scaled * price_std + price_mean

        # Convert the price to INR
        today_predicted_price_inr = today_predicted_price * EXCHANGE_RATE

        return render_template('result.html', result=today_predicted_price_inr)

if __name__ == '__main__':
    app.run(debug=True)
