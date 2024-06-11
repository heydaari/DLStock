import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
import datetime as dt

companies = []
company = input('ENTER THE NAME OF COMPANY : ')

while company != 'exit' :
    companies.append(company)
    company = input('ENTER THE NAME OF COMPANY : ')

yfin.pdr_override()

for item in companies :

    data = pdr.get_data_yahoo(item, start='2020-01-01', end=dt.datetime.now())
    file = data.to_csv(f'Data/{item}.csv')

print('DONE !')

