import pandas as pd
import datetime
from sqlalchemy import create_engine
import db
from db import DB
import psycopg2 as pg
import numpy as np
import sqlalchemy as sa
import os
import random
from string import digits

### LOGIN TO REDSHIFT

engine = create_engine('redshift+psycopg2://USERNAME:PASSWORD@DATABASE-NAME-ADDRESS:5439/prod')

def redshift():
	"""Connect to RedShift using environmental variables."""
	kwargs = {
			'user' : 'USERNAME',
			'password' : 'PASSWORD!',
			'host' : 'DATABASE-NAME-ADDRESS',
			'database' : 'prod',
			'port' : '5439',
			'client_encoding':'utf-8'
			}

	connection = pg.connect(**kwargs)
	return connection

conn = redshift()

cursor = conn.cursor()

db = db.DB(username="USERNAME",  password="PASSWORD", dbname="prod",
port="5439", hostname="DATABASE-NAME-ADDRESS",dbtype="redshift",exclude_system_tables=True)

### GET CORRECT TICKERS FROM CSV

company_list = pd.read_csv('~/acquisition-prediction/All_ticker_list.csv')

powers = {'B': 10 ** 9, 'M': 10 ** 6, 'T': 10 ** 12}
# add some more to powers as necessary

def f(s):
    try:
        s = s.replace(',','').replace(' ','').replace('$','')
        if s == 'n/a':
            return 0.0
        power = s[-1]
        return float(s[:-1]) * powers.get(power,'M')
    except TypeError:
        return float(s)
    
company_list['mcap'] = company_list['MarketCap'].map(f)

### FILTER DATA ACCORDING TO MARKET CAP
Market_Cap_Limit = 100000000.00
Market_Cap_Max = 100000000000.00

mask = (company_list['mcap'] > Market_Cap_Limit) & (company_list['mcap'] < Market_Cap_Max)
company_list_short = company_list[mask].sort_values(by=['mcap'],ascending=False)
company_symbols_short = list(company_list_short.Symbol)

acquisition_list = pd.read_csv('~/acquisition-prediction/acquisitions_predict.csv')
acquisition_tickers = list(acquisition_list['gfd_ticker'])

### CLEAN ACQUISITION TICKERS
remove_digits = str.maketrans('', '', digits)

acquisition_tickers_clean = []
for tick in acquisition_tickers:
    if tick == '1-May':
        tick = 'MAY1'
    tick = tick.translate(remove_digits)
    acquisition_tickers_clean.append(tick)
    
company_symbols_short = [tick for tick in company_symbols_short if tick not in acquisition_tickers_clean]
    
#### PREP STATEMENTS FOR DATA PULL
case_string = """(CASE 
"""
for i,tick in enumerate(acquisition_tickers):
    s = " WHEN b.option_ticker = '" + acquisition_tickers_clean[i] + "' THEN '" + tick + "' \n"
    case_string += s

case_string += "ELSE b.option_ticker END)"


case_string2 = """(CASE 
"""
for i,tick in enumerate(company_symbols_short):
    s = " WHEN b.option_ticker = '" + company_symbols_short[i].strip() + "' THEN '" + tick + "' \n"
    case_string2 += s

case_string2 += "ELSE b.option_ticker END)"

acq_tick_str = "('" + "','".join(acquisition_tickers) + "')"
clean_acq_tick_str = "('" + "','".join(acquisition_tickers_clean) + "')"
company_tick_str = "('" + "','".join([sym.strip() for sym in company_symbols_short]) + "')"

# PULL ACQUIRED COMPANY TICKERS

query1 = """ SELECT DISTINCT a.ticker as stock_ticker, 
            b.option_ticker,
            MIN(a.stockdate) OVER (PARTITION BY a.ticker) as stock_start,
            c.acquisition_date,
            MAX(a.stockdate) OVER (PARTITION BY a.ticker) as stock_end,
            COUNT(*) OVER (PARTITION BY a.ticker) as stocks_counted,
            COUNT(*) over (PARTITION BY b.option_ticker) as options_counted
            FROM stocks as a INNER JOIN """
query2 = """(SELECT DISTINCT ticker as option_ticker, data_date as options_date FROM options_wide 
                WHERE data_date >= '1/1/2002' AND ticker IN """ + clean_acq_tick_str
query3 = """) as b ON a.stockdate = b.options_date AND a.ticker = """ + case_string
query35 = """ LEFT JOIN acquisitions as c ON a.ticker = c.ticker """
query4 = """ WHERE a.stockdate >= '1/1/2002' AND a.ticker IN """ + acq_tick_str

query = query1 + query2 + query3 + query35 + query4
acq_ticker_counts = db.query(query)

acq_ticker_counts['acquisition_date'] = pd.to_datetime(acq_ticker_counts['acquisition_date'])
acq_ticker_counts['stock_start'] = pd.to_datetime(acq_ticker_counts['stock_start'])
acq_ticker_counts['stock_end'] = pd.to_datetime(acq_ticker_counts['stock_end'])
acq_ticker_counts['days_good_data'] = ((acq_ticker_counts['acquisition_date'] - acq_ticker_counts['stock_start']) / np.timedelta64(1, 'D')).astype(int)
mask = (acq_ticker_counts['stocks_counted'] >= 360) & (acq_ticker_counts['days_good_data'] >= 360)
acq_ticker_counts = acq_ticker_counts[mask]
acq_stock_tickers = acq_ticker_counts['stock_ticker'].tolist()
acq_stock_dates = acq_ticker_counts['acquisition_date'].tolist()
acq_options_tickers = acq_ticker_counts['option_ticker'].tolist()

# NON-ACQUIRED COMPANY TICKERS

query1 = """ SELECT DISTINCT a.ticker as stock_ticker, 
            b.option_ticker,
            MIN(a.stockdate) OVER (PARTITION BY a.ticker) as stock_start,
            c.acquisition_date,
            MAX(a.stockdate) OVER (PARTITION BY a.ticker) as stock_end,
            COUNT(*) OVER (PARTITION BY a.ticker) as stocks_counted,
            COUNT(*) over (PARTITION BY b.option_ticker) as options_counted
            FROM stocks as a INNER JOIN """
query2 = """(SELECT DISTINCT ticker as option_ticker, data_date as options_date FROM options_wide 
                WHERE data_date >= '1/1/2002' AND ticker IN """ + company_tick_str
query3 = """) as b ON a.stockdate = b.options_date AND a.ticker = b.option_ticker"""
query35 = """ LEFT JOIN acquisitions as c ON a.ticker = c.ticker """
query4 = """ WHERE a.stockdate >= '1/1/2002' AND a.ticker IN """ + company_tick_str

query = query1 + query2 + query3 + query35 + query4
existing_ticker_counts = db.query(query)

existing_ticker_counts['stock_start'] = pd.to_datetime(existing_ticker_counts['stock_start'])
existing_ticker_counts['stock_end'] = pd.to_datetime(existing_ticker_counts['stock_end'])
existing_ticker_counts['days_good_data'] = ((existing_ticker_counts['stock_end'] - existing_ticker_counts['stock_start']) / np.timedelta64(1, 'D')).astype(int)
mask = (existing_ticker_counts['stocks_counted'] >= 360) & (existing_ticker_counts['days_good_data'] >= 360)
existing_ticker_counts = existing_ticker_counts[mask]
exist_stock_tickers = existing_ticker_counts['stock_ticker'].tolist()
exist_option_tickers = existing_ticker_counts['option_ticker'].tolist()

# COMBINE TICKER LISTS
final_ticker_list = exist_stock_tickers + acq_stock_tickers
final_ticker_list = [tick.strip() for tick in final_ticker_list]
random.shuffle(final_ticker_list)
print(len(final_ticker_list))
print(final_ticker_list[:30])

final_tick_str = "('" + "','".join(final_ticker_list) + "')"


#### USE DATA TO PULL LIST
case_string = """(CASE 
"""
for i,tick in enumerate(acquisition_tickers):
    s = " WHEN b.ticker = '" + acquisition_tickers_clean[i] + "' THEN '" + tick + "' \n"
    case_string += s

case_string += "ELSE b.ticker END)"

today = datetime.datetime.now().strftime('%Y_%m_%d_%I%M%p')

final_stock_data = pd.DataFrame()

size = int(len(final_ticker_list)/16)
start = 0
end = size
for i in range(16):
    tick_str = "('" + "','".join(final_ticker_list[start:end]) + "')"
    query1 = """ SELECT DISTINCT
                        a.ticker,
                        a.stockdate as data_date,
                        a.volume,
                        a.close,
                        a.high,
                        a.low,
                        b.days_15_neg_price,
                        b.days_15_pos_price,
                        b.days_30_neg_price,
                        b.days_30_pos_price,
                        b.days_60_neg_price,
                        b.days_60_pos_price,
                        b.days_90_neg_price,
                        b.days_90_pos_price,
                        b.days_120_neg_price,
                        b.days_120_pos_price,
                        b.days_150_neg_price,
                        b.days_150_pos_price,
                        b.days_180_neg_price,
                        b.days_180_pos_price,
                        CASE
                        WHEN c.acquisition_date IS NULL THEN 0
                        WHEN DATEDIFF('d',a.stockdate, c.acquisition_date) <= 210 THEN 1
                        ELSE 0 END as acquisition_pending
                        FROM stocks as a
                        INNER JOIN options_wide as b ON a.stockdate = b.data_date AND a.ticker = """ + case_string 
    query2 = """ LEFT JOIN acquisitions as c ON a.ticker = c.ticker  
                WHERE a.stockdate >= '1/1/2002' 
                AND ((c.acquisition_date IS NULL ) OR (c.acquisition_date > a.stockdate))
                AND a.ticker IN """ + tick_str
    final_query = query1 + query2
    stock_data = db.query(final_query)
    print("Query Completed: ", tick_str)
    stock_data.sort_values(by=['ticker','data_date'],inplace=True)
    stock_data.drop_duplicates(inplace=True)

    final_stock_data = final_stock_data.append(stock_data, ignore_index=True)
    print("Data Appended: ", i)
    start += size
    end += size
    if i == 14:
        end = len(final_ticker_list)

today = datetime.datetime.now().strftime('%Y_%m_%d_%I%M%p')
final_stock_data.sort_values(by=['ticker','data_date'],inplace=True)
final_stock_data.drop_duplicates(inplace=True)
final_stock_data.to_csv('~/acquisition-prediction/final_stock_data.csv', index=False)


    

