import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
today = pd.to_datetime(date.today())
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None


def clean_data(bondData, sheet):
    today = pd.Timestamp.today()
    bondData = bondData[sheet]
    bondData.columns = bondData.iloc[0]
    bondData = bondData.drop([0])
    # We make time data readable
    bondData['Maturity Date'] = pd.to_datetime(bondData['Maturity Date'], errors='coerce')
    bondData['Coupon Last Date'] = pd.to_datetime(bondData['Coupon Last Date'], errors='coerce')
    bondData['Next Coupon Date'] = pd.to_datetime(bondData['Next Coupon Date'], errors='coerce')
    bondData['Issue Date'] = pd.to_datetime(bondData['Issue Date'], errors='coerce')
    bondData['Maturity Date'] = bondData['Maturity Date'].dt.strftime('%d-%m-%Y')
    bondData['Coupon Last Date'] = bondData['Coupon Last Date'].dt.strftime('%d-%m-%Y')
    bondData['Next Coupon Date'] = bondData['Next Coupon Date'].dt.strftime('%d-%m-%Y')
    bondData['Issue Date'] = bondData['Issue Date'].dt.strftime('%d-%m-%Y')
    bondData = bondData.drop(columns=['Ticker', 'Coupon Frequency', 'Price Accrued Interest Flag'])
    bondData['Ttm'] = ((pd.to_datetime(bondData['Maturity Date'], dayfirst=True, errors='coerce') - today).dt.days / 365).round(3)
    bondData['Coupon'] = bondData['Coupon'] / 2
    bondData['Coupon'] = pd.to_numeric(bondData['Coupon'], errors='coerce')
    bondData[~bondData['Yield to Maturity'].isna()]
    bondData['Yield to Maturity'] = bondData['Yield to Maturity'] / 100
    # Remove rows where column 'Dirty Price' is not a number
    bondData = bondData[~bondData['Dirty Price'].astype(str).str.contains('[a-zA-Z]', na=False)]
    # Drop first bond as it's usually volatile
    # bondData = bondData.iloc[1:].reset_index(drop=True)

    # Remove '/d' from description end
    bondData['Description'] = bondData['Description'].str.rstrip('/d')
    bondData['Description'] = bondData['Description'].str.replace(r'(\d{2})(\d{2})$', r'\1/20\2', regex=True)

    return bondData
    breakpoint()


def bond_cleaner(bondData):
    bondData['Maturity Date'] = pd.to_datetime(bondData["Maturity Date"], format='%d-%m-%Y', dayfirst=True)
    bondData['Issue Date'] = pd.to_datetime(bondData["Issue Date"], format='%d-%m-%Y',dayfirst=True)
    # Convert the 'Next Coupon Date' column to ensure consistent handling of NaT values
    bondData['Next Coupon Date'] = pd.to_datetime(bondData['Next Coupon Date'], format='%d-%m-%Y', errors='coerce')
    bondData['Next Coupon Date'] = bondData['Next Coupon Date'].replace(['None', 'NA', '', ' '], np.nan)
    today = pd.Timestamp.today()

    bondData['ISIN'] = bondData['ISIN'].apply(lambda x: 'IT' + ''.join([str(np.random.randint(0, 10)) for _ in range(10)]) if pd.isna(x) else x)

    return bondData