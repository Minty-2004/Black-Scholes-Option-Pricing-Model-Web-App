# Black-Scholes-Option-Pricing-Model-Web-App
This interactive Streamlit app allows users to dynamically explore and visualise call and put option prices using the Black-Scholes formula. Users can input a stock ticker, customise option parameters, and visualise results through a heatmap with optional profit/loss colouring.

---

## Features

- **Live stock data input** via ticker symbol
- **Black-Scholes pricing** for call and put options
- Customizable input parameters: stock price, strike, time to maturity, volatility, risk-free rate
- **Interactive heatmap**:
  - Choose **any two parameters** to plot option price sensitivity
  - Optional red/green **profit/loss visualisation** if purchase prices are entered

---

- **Python 3**
- `streamlit` for app visualisation
- `pandas` for data manipulation
- `yfinance` for downloading historical stock data
- `matplotlib and seaborn` for data visualisation
- `numpy` for numerical calculations
- `datetime` for creating time frames

---

## Usage

(Optional) Enter a stock ticker (e.g., AAPL)

1. Input or adjust the Black-Scholes parameters:
     Current stock price (S)
     Strike price (K)
     Time to maturity (T)
     Risk-free rate (r)
     Volatility (σ)

2. Select two variables to create a heatmap

(Optional) Enter a purchase price to see profit/loss zones

---

## How to Run

1. Clone this repository
2. Install dependencies (see requirements.txt)
3. Run the script (main.py)

---

## Demo Web App

> [Click here!](https://bscholes-heatmap-minty.streamlit.app/)

---

## Purpose

This project was created in the early steps of my journey into Quantitative Analysis with Python. It demonstrates:

- Financial data analysis with Python
- Black-Scholes model understanding and implementation
- Experimentation with the StreamLit Library

---

## Author

**Muhammad Muntasir Shahzad**  
Student at King's College London, University of London. Studying Mathematics with Management and Finance   
Graduating: Summer 2026  
[LinkedIn Profile](www.linkedin.com/in/muntasir-shahzad) | [Email](muntasir.s.2004@gmail.com)

Please don't hesitate to contact me if you have any questions, suggestions, or otherwise.

---

## Disclaimer

This code is for educational purposes only and does not constitute financial advice or an investment recommendation.
