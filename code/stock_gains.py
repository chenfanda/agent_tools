# filename: stock_gains.py
import yfinance as yf
from datetime import date

# define tickers and start of year
tickers = ['TSLA', 'META']
start_of_year = date(date.today().year, 1, 1)

# fetch data from yfinance
data = yf.download(tickers, start=start_of_year)

# calculate year to date gain and print results
ytd_gain = ((data['Close'] - data.loc[start_of_year].values)[-1] * 100).round(2)
print("Year-to-Date Gains:")
for ticker, gain in zip(tickers, ytd_gain):
    print(f"{ticker}: {gain}%")