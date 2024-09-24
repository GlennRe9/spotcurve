import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import date
import json
import statistics
import re
import math
from datetime import date, timedelta
import scipy.interpolate as inter
from scipy.interpolate import UnivariateSpline
today = pd.to_datetime(date.today())
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)

pd.options.mode.chained_assignment = None


#   The following programme extracts from a list of italian BTPs and creates a new list of cashflows made by all coupons
#   and notional payments from each bond of the list. It then uses a spline interpolation based on the market yield to
#   maturities of zero-coupon bonds from the list to create a spot curve for all bonds up to the last zero-coupon bond
#   present on the italian curv (currently the IT0005454241 08/26). For cashflows after this maturity, a standard bootstripping
#   method is used.

#Useful resources: https://quant.stackexchange.com/questions/3302/deriving-spot-rates-from-treasury-yield-curve


def curve_visualiser(spline, x, y, x1, x2):
    myX = np.linspace(x1, x2, 100)
    plt.plot(myX, spline(myX), label='firsthalf')
    plt.axis([x1, x2, -0.01, 0.04])
    plt.scatter(x, y)
    plt.show()
def clean_data(bondData):
    today = pd.Timestamp.today()
    bondData = bondData['Monitor']
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
    bondData = bondData.drop(columns=['Ticker'])
    bondData['Ttm'] = ((pd.to_datetime(bondData['Maturity Date'], dayfirst=True, errors='coerce') - today).dt.days / 365).round(3)
    bondData['Coupon'] = bondData['Coupon']
    bondData['Yield to Maturity'] = bondData['Yield to Maturity'] / 100

    # Drop first bond as it's usually volatile
    # bondData = bondData.iloc[1:].reset_index(drop=True)

    # Remove '/d' from description end
    bondData['Description'] = bondData['Description'].str.rstrip('/d')
    bondData['Description'] = bondData['Description'].str.replace(r'(\d{2})(\d{2})$', r'\1/20\2', regex=True)

    return bondData
    breakpoint()

def couponCalculator(bond,bondData):

    maturity = pd.to_datetime(bond["Maturity Date"], dayfirst=True)
    issue_date = pd.to_datetime(bond["Issue Date"], dayfirst=True)
    today = pd.Timestamp.today()
    coupon = bond['Coupon']

    cashflow_dates = []
    next_coupon_date = maturity
    while next_coupon_date >= issue_date:
        cashflow_dates.append(next_coupon_date)
        next_coupon_date -= pd.DateOffset(months=6)
    cashflow_dates = pd.DatetimeIndex(cashflow_dates)
    cashflow_dates = cashflow_dates[cashflow_dates >= today]

    cashflow_dates = pd.DatetimeIndex(cashflow_dates)

    # Convert to a DataFrame for better visualization (optional)
    cashflow_df = pd.DataFrame({
        'ISIN': bond['ISIN'],
        'Description': bond['Description'],
        'CF Date': cashflow_dates,
        'Dirty Price': bond['Dirty Price'],
        'Coupon': coupon,
        'Maturity Date': maturity,
        'Yield to Maturity': bond['Yield to Maturity'],
        'Ttm': bond['Ttm'],
        'Spot': None
    })

    cashflow_df.loc[cashflow_df['CF Date'] == maturity, 'Coupon'] = str(float(coupon) + 100)
    cashflow_df['Ttm'] = ((cashflow_df['CF Date'] - today).dt.days / 365).round(3)

    # If no ISIN, we generate a random one for cleanness
    if cashflow_df['ISIN'].isna().any():
        new_isin = 'IT' + ''.join([str(np.random.randint(0, 10)) for _ in range(10)])
        cashflow_df['ISIN'] = new_isin
    return cashflow_df




def curve_builder(bondDatabase, cashflow_df):

    # Pure zeros are actual zero-coupon bonds
    # Non-Pure zeroes are coupon bonds which only have one cashflow left
    pure_zeros = bondDatabase[(bondDatabase['Coupon'] == 0) & (pd.notna(bondDatabase['Next Coupon Date']))]
    logging.info(f'Number of zero-coupon bonds: {len(pure_zeros)}')
    coupon_zeros = bondDatabase[(pd.isna(bondDatabase['Next Coupon Date'])) & (bondDatabase['Coupon'] != 0)]
    coupon_zeros['Spot'] = coupon_zeros['Yield to Maturity']
    logging.info(f'Number of coupon zeros: {len(coupon_zeros)}')


    cashflow_df = cashflow_df.sort_values(by='CF Date').reset_index(drop=True)

    # We define the coupon zero spline
    spline = inter.InterpolatedUnivariateSpline(coupon_zeros['Ttm'], coupon_zeros['Yield to Maturity'])
    regSpline = inter.UnivariateSpline(coupon_zeros['Ttm'], coupon_zeros['Yield to Maturity'])

    #curve_visualiser(regSpline, coupon_zeros['Ttm'], coupon_zeros['Yield to Maturity'], 0, 5)

    # We take all the cashflows that are less than the maximum time to maturity of the coupon_zeros, and interpolate them
    # With the spline from the coupon_zeros

    # In general, this approach is ok. A more refined approach would be to only do so for the bonds that have one expiry in
    # that timeframe. In fact, if we had a bond with two final cashflows in that horizon, we could interpolate the coupon cf
    # and bootstrap the maturity cf.

    max_ttm = coupon_zeros['Ttm'].max()
    mask = cashflow_df['Ttm'] < max_ttm
    cashflow_df.loc[mask, 'Spot'] = regSpline(cashflow_df.loc[mask, 'Ttm'].values)

    # We now have a base of spot rates. Next, for every cashflow that is comes from a bond with more than one cashflow
    # outside the interpolated area, we interpolate and find its spot rate. For every bond that only has one cashflow,
    # we bootstrap.

    #While doing so, we recalculate the interpolation.
    regSpline = inter.UnivariateSpline(cashflow_df['Ttm'], cashflow_df['Spot'])

    missing_spots = cashflow_df[cashflow_df['Spot'].isna()]

    for isin in missing_spots['ISIN'].unique():
        bond_cf = cashflow_df[cashflow_df['ISIN'] == isin]

        if len(bond_cf) == 1:
            bootstrap_spot(bond_cf, regSpline)
        else:
            # Take that cashflow from cashflow_df and interpolate to get the spot
            first_cf = bond_cf[bond_cf['Spot'].isna()].index[0]
            cashflow_df.loc[first_cf, 'Spot'] = regSpline(cashflow_df.loc[first_cf, 'Ttm'])

        # We rerun the spline in order to take into account the latest spots
        regSpline = inter.UnivariateSpline(cashflow_df['Ttm'], cashflow_df['Spot'])

        breakpoint()




def bootstrap_spot(bond_cf, regSpline):
        breakpoint()
        nCFs = len(bond_cf)
        finalCashFlow = bond_cf.iloc[nCFs - 1]['Coupon']
        maturity = bond_cf.iloc[nCFs - 1]['Ttm']
        dirtyPrice = bond_cf.iloc[nCFs - 1]['Dirty Price']
        sumDiscCFs = 0
        for i in range(0, nCFs - 1):
            period = bond_cf.iloc[i]['Ttm']
            sumDiscCFs += (bond_cf.iloc[i]['Coupon']) / (1 + bond_cf.iloc[i]['Spot']) ** (bond_cf.iloc[i]['Ttm'])
        spot = ((finalCashFlow / (dirtyPrice - sumDiscCFs)) ** (1 / (maturity)) - 1)
        return spot

def main():


    bondData = pd.read_excel("BTP_data.xlsx", parse_dates=True, sheet_name=['Monitor'])
    bondData = clean_data(bondData)

    bondDatabase = bondData.copy()

    bondData = bondData[:30]


    cashflow_list = bondData.apply(lambda row: couponCalculator(row,bondData), axis=1)
    cashflows_df = pd.concat(cashflow_list.tolist(), ignore_index=True)


    curve_builder(bondDatabase,cashflows_df)





main()