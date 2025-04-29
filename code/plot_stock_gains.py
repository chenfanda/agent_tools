# filename: plot_stock_gains.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Manually set backend to avoid environment issues
from matplotlib import use
use('Agg')  # Use 'Agg' for headless environments

# Start and end date (YTD through today 2024-11-20)
start_date = '2023-01-01'
end_date = '2024-11-20'

# Tickers for Tesla and Meta
tickers = ['TSLA', 'META']

# Download the data
stocks_data = {ticker: yf.download(ticker, start=start_date, end=end_date) for ticker in tickers}

# Gain calculation and plot setup
plt.figure(figsize=(14, 7))

for ticker, data in stocks_data.items():
    # Calculate gain from opening price of year to each day's closing price.
    first_day_open = data['Open'][0]
    gains = (data['Close'] - first_day_open) / first_day_open * 100
    plt.plot(gains.index, gains.values, label=ticker)

# Plotting details
plt.title('YTD Stock Gain: TSLA vs META')
plt.xlabel('Date')
plt.ylabel('% Gain YTD')
plt.legend()
plt.grid(True)
plt.savefig('stock_gains.png')  # Save the plot

print("Plot 'stock_gains.png' has been saved.")