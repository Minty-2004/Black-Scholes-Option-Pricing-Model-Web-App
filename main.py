from scipy.stats import norm
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs
import time
from datetime import datetime,date,timedelta
import seaborn as sns
from  matplotlib.colors import LinearSegmentedColormap

#Page config
st.set_page_config(layout="wide")
st.title("Black-Scholes Pricing Model")
with st.sidebar:
    st.subheader("Created by **Muhammad Muntasir Shahzad**")
    with st.expander("About this app", expanded=False):
        st.write("This app calculates the price of call and put options using the Black-Scholes model.")
        st.write("You can use realistic stock data to extract some parameters, or set your own parameters entirely.")
        st.write("The app also provides a heatmap to visualize how the option prices change with different parameters. This can be configured in the sidebar, and can visualise profits/losses if purchase prices are entered.")
        st.write("If you have any questions or suggestions, feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/muntasir-shahzad/)")
        st.write("You can find the source code on [GitHub](https://github.com/Minty-2004/Black-Scholes-Option-Pricing-Model-Web-App)")
    st.divider()


# Setting parameters for the model

# Realistic stock data toggle
realistic = st.sidebar.toggle("Use realistic stock data")
if realistic:
    with st.sidebar:
        st.write("Select the stock for which you want to calculate the option price")
        ticker = st.text_input("Ticker", value="AAPL")
        data = yf.Ticker(ticker)
        recent_data = data.history(period="1y")
        recent_close = pd.Series(recent_data["Close"])
        std = qs.stats.volatility(recent_close)
        rf_data = yf.Ticker("^TNX")
        rf = rf_data.info["regularMarketPrice"]
        S = st.sidebar.number_input("Stock price (current, in $", value=data.info["currentPrice"], step=0.10, format="%0.2f", disabled=True)
        K = st.sidebar.number_input("Strike price, in $", value=S*1.05, step=0.10, format="%0.2f", min_value=0.00)
        T = st.sidebar.number_input("Time to expiration (years)", value=0.5, step=0.5, min_value=0.0)
        r = st.sidebar.number_input("Risk-free interest rate in % - 10-Year US Treasury Bond Yield", value=rf, step=0.001, format = "%0.3f", disabled=True)
        vol = st.sidebar.number_input("Volatility of stock (σ)", value=std, step=0.01, disabled=True)
        st.divider()
else:
    # Manual input for parameters
    with st.sidebar:
        st.sidebar.write("Select the parameters for the model")
        S = st.sidebar.number_input("Stock price in $", value=50.0, step=0.1, format="%0.2f", min_value=0.0)
        K = st.sidebar.number_input("Strike price in $", value=52.0, step=0.1, format="%0.2f", min_value=0.0)
        T = st.sidebar.number_input("Time to expiration (years)", value=0.5, step=0.5, min_value=0.0)
        r = st.sidebar.number_input("Risk-free interest rate in % (annualised)", value=5.000, step=0.001, format="%0.3f", min_value=0.000, max_value=100.000)
        vol = st.sidebar.number_input("Volatility (σ)", value=0.10, step=0.01, format="%0.2f", min_value=0.00)
        st.divider()

# Displaying the parameters as a dictionary
st.subheader("Parameters for the model:")
model_parameters = {
    "Stock price ($)": [round(S,2)],
    "Strike price ($)": [round(K,2)],
    "Time to Expiration (years)": [T],
    "Risk-free Rate (%)": [round(r,3)],
    "Volatility": [round(vol,2)]
}
params= pd.DataFrame.from_dict(model_parameters)
st.write(params)

#Black-Scholes model calculations
def black_scholes(S,K,T,r,vol):
    r = r/100
    if vol == 0 or T == 0:
        epsilon=1e-10
        d1 = (np.log(S/K)+(r+(0.5*(vol**2)))*T)/((vol*np.sqrt(T))+epsilon)
    else:
        d1 = (np.log(S/K)+(r+(0.5*(vol**2)))*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    call = norm.cdf(d1)*S - norm.cdf(d2)*K*np.exp(-r*T)
    put = norm.cdf(-d2)*K*np.exp(-r*T) - norm.cdf(-d1)*S
    return(call,put)
call_price, put_price = black_scholes(S,K,T,r,vol)

# Style for output containers
greencont = """
.st-key-green_container {
    background-color: rgba(0, 255, 127, 0.8);
    text-color: black;
    text-align: center;
}
"""
st.html(f"<style>{greencont}</style>")

redcont = """
.st-key-red_container {
    background-color: rgba(240, 128, 128, 2);
    text-color: black;
}
"""
st.html(f"<style>{redcont}</style>")

#Outputting results
c1,c2 = st.columns(2)
with c1:
    with st.container(key="green_container", border=True):
        st.write('<p style="font-size:14px; color:black; text-align:center; margin-bottom:0;">CALL Value</p>'  
                 '<p style="font-size:28px; color:black; text-align:center; margin-top:0; font-weight:bold;">${}</p>'.format(round(call_price,2))
                 ,unsafe_allow_html=True)
with c2:
    with st.container(key="red_container", border=True):
        st.write('<p style="font-size:14px; color:black; text-align:center; margin-bottom:0;">PUT Value</p>'  
                 '<p style="font-size:28px; color:black; text-align:center; margin-top:0; font-weight:bold;">${}</p>'.format(round(put_price,2))
                 ,unsafe_allow_html=True)

#Heatmap Parameters

with st.sidebar:
    st.write('<p style="font-size:20px; text-align:left; margin-top:0; font-weight:bold;">Heatmap Settings</p>', unsafe_allow_html=True)
    #Selecting two parameters for the heatmap
    heatmap_params = st.multiselect("Pick two parameters to analyse in the heatmap", options=model_parameters, max_selections=2)

    #For each selected parameter, create a slider to select the range
    heatmap_ranges = []
    for str in heatmap_params:
        if str == "Stock price ($)":
            S_hmr = st.slider("Select the Stock price range ($)",min_value=0.0,max_value=2*S,value=[S/2,3*S/2],step=0.1,format="%0.1f")
            heatmap_ranges.append([str,f"{round(S_hmr[0],1)} - {round(S_hmr[1],1)}"])
        elif str == "Strike price ($)":
            K_hmr = st.slider("Select the Strike price range ($)",min_value=0.0,max_value=2*K,value=[K/2,3*K/2],step=0.1,format="%0.1f")
            heatmap_ranges.append([str,f"{round(K_hmr[0],1)} - {round(K_hmr[1],1)}"])
        elif str == "Time to Expiration (years)":
            T_hmr = st.slider("Select the time range (years)",min_value=0.0,max_value=10.0,value=[0.0,T],step=0.1,format="%0.1f")
            heatmap_ranges.append([str,f"{round(T_hmr[0],1)} - {round(T_hmr[1],1)}"])
        elif str == "Risk-free Rate (%)":
            r_hmr = st.slider("Select the range for the Risk-Free Rate (%)",min_value=0.00,max_value=100.00,value=[r/2,3*r/2],step=0.01,format="%0.2f")
            heatmap_ranges.append([str,f"{round(r_hmr[0],3)} - {round(r_hmr[1],3)}"])
        elif str == "Volatility":
            vol_hmr = st.slider("Select the Volatility range",min_value=0.00,max_value=5.00,value=[0.00,1.00],step=0.01,format="%0.2f")
            heatmap_ranges.append([str,f"{round(vol_hmr[0],2)} - {round(vol_hmr[1],2)}"])

    #Creating a dictionary and pandas series for heatmap ranges, allows data to be extracted easily
    heatmap_ranges_dict = {key: value for key, value in heatmap_ranges}
    heatmap_ranges_pd = pd.Series(data=heatmap_ranges_dict,index=None,name="Heatmap Ranges")

    #Option for P&L heatmap
    selected = st.checkbox("Enter option purchase prices (for P&L heatmap)?")
if selected:
    call_purchase = st.sidebar.number_input("Enter purchase price for the call, in $", value=call_price, step=0.10, min_value=0.00, format="%0.2f")
    put_purchase = st.sidebar.number_input("Enter purchase price for the put, in $", value=put_price, step=0.10, format="%0.2f", min_value=0.00)
    
    #Introducing colormap for P&L heatmap
    c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
    v = [0,.15,.4,.5,0.6,.9,1.]
    l = list(zip(v,c))
    cmap=LinearSegmentedColormap.from_list('rg',l)

# Heatmap not generated if less than 2 parameters selected
if len(heatmap_params) == 2: 
    st.divider()
    #Heatmap range divisions + setting up heatmap
    p1_full = heatmap_ranges[0][0]
    p1_range = heatmap_ranges[0][1]
    p1_str_endpoints = p1_range.split(" - ")
    p1_e1 = float(p1_str_endpoints[0])
    p1_e2 = float(p1_str_endpoints[1])
    if p1_e1 == p1_e2:
        with st.sidebar:
            st.error("Sorry, the range needs to be greater than 0. Please edit and try again.")
            st.stop()
    
    p2_full = heatmap_ranges[1][0]
    p2_range = heatmap_ranges[1][1]
    p2_str_endpoints = p2_range.split(" - ")
    p2_e1 = float(p2_str_endpoints[0])
    p2_e2 = float(p2_str_endpoints[1])
    if p2_e1 == p2_e2:
        with st.sidebar:
            st.error("Sorry, the range needs to be greater than 0. Please edit and try again.")
            st.stop()

    p1_divided_range=[]
    p1_diff = (p1_e2-p1_e1)/9
    for i in range (10):
        p1_divided_range.append(round(p1_e1+(i*p1_diff),2))

    p2_divided_range=[]
    p2_diff = (p2_e2-p2_e1)/9
    for i in range (10):
        p2_divided_range.append(round(p2_e1+(i*p2_diff),2))
    p2_divided_range.reverse()

    #Creating data grids for heatmap, using numpy meshgrid to create a grid of user parameter values and ensuring every combination is calculated
    X,Y = np.meshgrid(p1_divided_range,p2_divided_range)
    call_heatmap = np.zeros_like(X)
    put_heatmap = np.zeros_like(X)
    
    #Calculating the call and put prices P&L for each combination of parameters
    if selected:
        for i in range (X.shape[0]):
            for j in range (X.shape[1]):
                hmparams = {
                    "Stock price ($)": round(S,2),
                    "Strike price ($)": round(K,2),
                    "Time to Expiration (years)": T,
                    "Risk-free Rate (%)": round(r,2),
                    "Volatility": round(vol,2)
                }
                hmparams[p1_full] = X[i,j]
                hmparams[p2_full] = Y[i,j]
                c,p = black_scholes(hmparams["Stock price ($)"],hmparams["Strike price ($)"],hmparams["Time to Expiration (years)"],hmparams["Risk-free Rate (%)"],hmparams["Volatility"])
                call_heatmap[i][j] = c - call_purchase
                put_heatmap[i][j] = p - put_purchase
    else:
        #Calculating the call and put prices for each combination of parameters
        for i in range (X.shape[0]):
            for j in range (X.shape[1]):
                hmparams = {
                    "Stock price ($)": round(S,2),
                    "Strike price ($)": round(K,2),
                    "Time to Expiration (years)": T,
                    "Risk-free Rate (%)": round(r,2),
                    "Volatility": round(vol,2)
                }
                hmparams[p1_full] = X[i,j]
                hmparams[p2_full] = Y[i,j]
                call_heatmap[i][j],put_heatmap[i][j] = black_scholes(hmparams["Stock price ($)"],hmparams["Strike price ($)"],hmparams["Time to Expiration (years)"],hmparams["Risk-free Rate (%)"],hmparams["Volatility"])

    #Using pandas to create dataframes for the heatmap, renaming the columns to the divided ranges for better heatmap readability
    call_heatmap_df = pd.DataFrame(call_heatmap, index=p2_divided_range)
    put_heatmap_df = pd.DataFrame(put_heatmap, index=p2_divided_range)
    for i in range (10):
        call_heatmap_df.rename(columns={i: p1_divided_range[i]}, inplace=True)
        put_heatmap_df.rename(columns={i: p1_divided_range[i]}, inplace=True)
    
    #Heatmap plotting
    c3,c4 = st.columns(2)
    with c3:
        if selected:
            st.subheader("Call Price Heatmap P&L")
        else:
            st.subheader("Call Price Heatmap")
        fig,ax = plt.subplots()
        if selected:
            #Ensuring the heatmap is centered around 0 for P&L, to allow for red=loss and green=profit
            extremum = max(np.abs(call_heatmap_df.min().min()),np.abs(call_heatmap_df.max().max()))
            vmin = -extremum
            vmax = extremum
            sns.heatmap(call_heatmap_df,annot=True, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".1f", ax=ax)
        else:
            sns.heatmap(call_heatmap_df,annot=True, cmap='plasma', fmt=".1f", ax=ax)
        plt.title("CALL")
        plt.xlabel(p1_full)
        plt.ylabel(p2_full)
        st.pyplot(fig, use_container_width=True)
    with c4:
        if selected:
            st.subheader("Put Price Heatmap P&L")
        else:
            st.subheader("Put Price Heatmap")
        fig,ax = plt.subplots()
        if selected:
            #Ensuring the heatmap is centered around 0 for P&L, to allow for red=loss and green=profit
            extremum = max(np.abs(put_heatmap_df.min().min()),np.abs(put_heatmap_df.max().max()))
            vmin = -extremum
            vmax = extremum
            sns.heatmap(put_heatmap_df,annot=True, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".1f", ax=ax)
        else:
            sns.heatmap(put_heatmap_df,annot=True, cmap='plasma', fmt=".1f", ax=ax)
        plt.title("PUT")
        plt.xlabel(p1_full)
        plt.ylabel(p2_full)
        st.pyplot(fig, use_container_width=True)