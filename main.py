import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import date
import json
import statistics
import re
import constraints as smoothing
import optimisation as optimisation
from datetime import date, timedelta
import scipy.interpolate as inter
from scipy.interpolate import UnivariateSpline
today = pd.to_datetime(date.today())
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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
    bondData['Coupon'] = bondData['Coupon'] / 2
    bondData[~bondData['Yield to Maturity'].isna()]
    bondData['Yield to Maturity'] = bondData['Yield to Maturity'] / 100

    # Drop first bond as it's usually volatile
    # bondData = bondData.iloc[1:].reset_index(drop=True)

    # Remove '/d' from description end
    bondData['Description'] = bondData['Description'].str.rstrip('/d')
    bondData['Description'] = bondData['Description'].str.replace(r'(\d{2})(\d{2})$', r'\1/20\2', regex=True)

    return bondData
    breakpoint()


def bond_cleaner(bondData):
    bondData['Maturity Date'] = pd.to_datetime(bondData["Maturity Date"], dayfirst=True)
    bondData['Issue Date'] = pd.to_datetime(bondData["Issue Date"], dayfirst=True)
    # Convert the 'Next Coupon Date' column to ensure consistent handling of NaT values
    bondData['Next Coupon Date'] = pd.to_datetime(bondData['Next Coupon Date'], errors='coerce')
    bondData['Next Coupon Date'] = bondData['Next Coupon Date'].replace(['None', 'NA', '', ' '], np.nan)
    today = pd.Timestamp.today()

    bondData['ISIN'] = bondData['ISIN'].apply(lambda x: 'IT' + ''.join([str(np.random.randint(0, 10)) for _ in range(10)]) if pd.isna(x) else x)

    return bondData

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


def driver(spot_df, coupon_bonds, cashflow_df):
    """
    Initiate the programme
    """
    """
    Iteratively handle each coupon bond, bootstrap its spot rate, perturb it,
    and adjust the forward curve.
    """
    # Start with zero-coupon bonds (cf_data is a copy of coupon_zeros)
    cf_data = spot_df.copy()

    # We use regspline here to calculate the cashflow yields
    regSpline = inter.UnivariateSpline(spot_df['Ttm'], spot_df['Yield to Maturity'], s=0.1)

    # Store estimated spot rates for the zero-coupon bonds
    estimated_spot_rates = list(spot_df['Spot'])

    # Loop through each coupon bond iteratively
    for i, coupon_bond in coupon_bonds.iterrows():
        logger.info(f"Processing coupon bond {i + 1} with maturity {coupon_bond['Ttm']} years.")

        # Step 1: Bootstrap the initial spot rate for the current coupon bond
        isin = coupon_bond['ISIN']
        des = coupon_bond['Description']
        bond_cf = cashflow_df[cashflow_df['ISIN'] == isin]
        bond_cf.sort_values(by='Ttm', inplace=True, ignore_index=True)

        # Step 2: Exclude the final cash flow and apply the spline to earlier maturities
        final_cf = bond_cf.iloc[-1]  # Last cash flow
        earlier_cfs = bond_cf.iloc[:-1]  # All cash flows except the last one
        max_ttm = earlier_cfs['Ttm'].max()
        mask = earlier_cfs['Ttm'] <= max_ttm

        # Apply the spline to get the spot rates for earlier cash flows
        earlier_cfs.loc[mask, 'Spot'] = regSpline(earlier_cfs.loc[mask, 'Ttm'].values)
        earlier_cfs['df'] = (1 / (1 + earlier_cfs['Spot'])) ** earlier_cfs['Ttm']
        bond_cf['df'] = (1 / (1 + bond_cf['Spot'])) ** bond_cf['Ttm']

        # Bootstrapping
        initial_spot_rate, initial_df = bootstrap_spot(isin, bond_cf, regSpline)

        logger.info(f"Initial spot rate for coupon bond {i + 1}: {initial_spot_rate:.6f}")
        # Update the 'Spot' for the last cash flow (final_cf) after bootstrapping
        bond_cf.loc[bond_cf.index[-1], 'Spot'] = initial_spot_rate
        bond_cf.loc[bond_cf.index[-1], 'df'] = initial_df

        final_cf = bond_cf.iloc[[-1]]  # Last cash flow
        final_cf.drop(columns=['CF Date'], inplace=True)

        # Step 2: Append this new spot rate to coupon_zeros
        updated_spot_df = pd.concat([spot_df, final_cf]).sort_values(by='Ttm').reset_index(drop=True)
        #updated_spot_df = pd.concat([updated_spot_df[:2], updated_spot_df[-2:]])

        x = optimisation.run_optimisation(updated_spot_df)
        segment_boundaries = updated_spot_df['Ttm'].values

        f = extract_last_forward(x, segment_boundaries,updated_spot_df)
        logger.info(f"First estimate of the forward rate for the coupon bond {des}: {f:.6f}")
        logger.info(f"First estimate of the discount rate for the coupon bond {des}: {initial_df}")


        # We update with the PQPI method
        updated_spot_df.loc[updated_spot_df['ISIN'] == isin, 'f'] = f

        updated_spot_df = perturbation_function(updated_spot_df)
        breakpoint()


        #estimated_px =

        build_forward_curve_from_coefficients(x, segment_boundaries)
        breakpoint()
        # # Call this after optimisation to evaluate or plot the forward curve
        # optimized_X = optimisation.run_optimisation(coupon_zeros)
        # forward_curve = build_forward_curve_from_coefficients(optimized_X, segment_boundaries)

        breakpoint()

def perturbation_function(updated_spot_df):
    s = updated_spot_df['Spot'].iloc[-1]
    Ttm = updated_spot_df['Ttm'].iloc[-1]
    isin = updated_spot_df['ISIN'].iloc[-1]
    act_price = updated_spot_df['Dirty Price'].iloc[-1] / 100
    est_price = updated_spot_df['df'].iloc[-1]

    while True:
        # Perturbation
        s_u = s + s / 100
        s_d = s - s / 100
        df_u = (1 / (1 + s_u)) ** Ttm
        df_d = (1 / (1 + s_d)) ** Ttm
        f_u = df_u / updated_spot_df['df'].iloc[-2]
        f_d = df_d / updated_spot_df['df'].iloc[-2]
        p_u = df_u
        p_d = df_d

        deriv = (p_u - p_d) / (s / 50)
        new_s = s + deriv * (act_price - est_price)
        new_df = (1 / (1 + new_s)) ** Ttm

        updated_spot_df.loc[updated_spot_df.index[-1], 'Spot'] = new_s
        updated_spot_df.loc[updated_spot_df.index[-1], 'df'] = new_df

        # Calculate spot difference for convergence
        spot_diff = abs(s - new_s)

        # Break if the condition is met
        if spot_diff <= 1e-9:
            break

        # Update the forward curve using PQPI optimisation after each perturbation
        x = optimisation.run_optimisation(updated_spot_df)
        segment_boundaries = updated_spot_df['Ttm'].values
        f = extract_last_forward(x, segment_boundaries, updated_spot_df)

        # Log the updated forward rate and update the DataFrame
        logger.info(f"forward rate being updated to {f}")
        updated_spot_df.loc[updated_spot_df['ISIN'] == isin, 'f'] = f

        # Update for the next iteration
        s = new_s
        est_price = new_df  # Update estimated price for the next perturbation loop

    return updated_spot_df
    # # Perturbation
    # s_u = s + s / 100
    # s_d = s - s / 100
    # df_u = (1 / (1 + s_u)) ** Ttm
    # df_d = (1 / (1 + s_d)) ** Ttm
    # f_u = df_u / updated_spot_df['df'].iloc[-2]
    # f_d = df_d / updated_spot_df['df'].iloc[-2]
    # p_u = df_u
    # p_d = df_d
    #
    # deriv = ( p_u - p_d) / (s / 50)
    # new_s = s + deriv * (act_price - est_price )
    # new_df = (1 / (1 + new_s)) ** Ttm
    #
    # updated_spot_df.loc[updated_spot_df.index[-1], 'Spot'] = new_s
    # updated_spot_df.loc[updated_spot_df.index[-1], 'df'] = new_df
    #
    #
    # spot_diff = abs(s - new_s)
    #
    # while spot_diff > 1e-9:
    #     x = optimisation.run_optimisation(updated_spot_df)
    #     segment_boundaries = updated_spot_df['Ttm'].values
    #
    #     f = extract_last_forward(x, segment_boundaries, updated_spot_df)
    #     logger.info(f"forward rate being updated to {f}")
    #     # We update with the PQPI method
    #     updated_spot_df.loc[updated_spot_df['ISIN'] == isin, 'f'] = f
    #     pertubation_function(updated_spot_df)
    #
    #
    # return updated_spot_df






    breakpoint()


def extract_last_forward(X, segment_boundaries,updated_spot_df):
    # We take the last Ttm
    last_t = updated_spot_df['Ttm'].iloc[-1]
    # We take the poly and calculate the forward rate
    a, b, c, d, e = X[-5:]
    f = a * last_t ** 4 + b * last_t ** 3 + c * last_t ** 2 + d * last_t + e

    return f


# Function to create and evaluate the forward curve
def build_forward_curve_from_coefficients(X, segment_boundaries):
    n_segments = len(segment_boundaries)
    nodes = n_segments - 1
    coefficients = X.reshape((n_segments, 5))  # Reshape x into 3 segments, 5 coefficients each

    # Define colors for each segment
    colors = ['blue', 'orange', 'yellow', 'green', 'red', 'black']

    # Plot each segment
    plt.figure(figsize=(10, 6))
    for i in range(nodes):  # We only plot the first two segments
        T_start, T_end = segment_boundaries[i], segment_boundaries[i + 1]
        a, b, c, d, e = coefficients[i]

        # Generate points for the segment
        T_values = np.linspace(T_start, T_end, 100)
        f_values = a * T_values ** 4 + b * T_values ** 3 + c * T_values ** 2 + d * T_values + e

        # Plot the segment
        plt.plot(T_values, f_values, color=colors[i], label=f'Segment {i + 1} ({T_start} to {T_end})')

    # Add labels and legend
    plt.xlabel("Time to Maturity (T)")
    plt.ylabel("Forward Rate f(T)")
    plt.title("Forward Rate Curve Segmented by Maturities")
    plt.legend()
    plt.grid(True)
    plt.show()


def curve_builder(bondDatabase, cashflow_df):

    # Pure zeros are actual zero-coupon bonds
    # Non-Pure zeroes are coupon bonds which only have one cashflow left
    pure_zeros = bondDatabase[(bondDatabase['Coupon'] == 0) & (bondDatabase['Next Coupon Date'].isna())]
    logging.info(f'Number of zero-coupon bonds: {len(pure_zeros)}')
    coupon_zeros = bondDatabase[(pd.isna(bondDatabase['Next Coupon Date'])) & (bondDatabase['Coupon'] != 0)]
    logger.info(f'Number of coupon zeros: {len(coupon_zeros)}')

    zeros = pd.concat([pure_zeros, coupon_zeros], axis=0)
    zeros['Spot'] = zeros['Yield to Maturity']
    zeros.sort_values(by='Ttm', inplace=True)

    zeros['df'] = (1 / (1 + zeros['Spot'])) ** zeros['Ttm']
    # divide zero['df'] by its previous value to get the forward rate
    zeros['f'] = (zeros['df'].shift(1) / zeros['df']) - 1


    # coupon bonds is the the bondDatabase where the isin is not in zeros
    coupon_bonds = bondDatabase[~bondDatabase['ISIN'].isin(zeros['ISIN'])].reset_index(drop=True)
    assert(len(zeros) + len(coupon_bonds) == len(bondDatabase))

    driver(zeros, coupon_bonds, cashflow_df)
    optimisation.iterative_optimisation(zeros, coupon_bonds)


    result = optimisation.run_optimisation(coupon_zeros)

    cashflow_df = cashflow_df.sort_values(by='CF Date').reset_index(drop=True)
    merged_df = cashflow_df.merge(coupon_zeros[['ISIN', 'Spot']], on='ISIN', how='left', suffixes=('', '_from_coupon'))
    cashflow_df['Spot'] = merged_df['Spot_from_coupon'].combine_first(cashflow_df['Spot'])

    # We define the coupon zero spline
    spline = inter.InterpolatedUnivariateSpline(coupon_zeros['Ttm'], coupon_zeros['Yield to Maturity'])
    regSpline = inter.UnivariateSpline(coupon_zeros['Ttm'], coupon_zeros['Yield to Maturity'], s=0.1)

    #curve_visualiser(regSpline, coupon_zeros['Ttm'], coupon_zeros['Yield to Maturity'], 0, 5)

    # We take all the cashflows that are less than the maximum time to maturity of the coupon_zeros, and interpolate them
    # With the spline from the coupon_zeros
    # In general, this approach is ok. A more refined approach would be to only do so for the bonds that have one expiry in
    # that timeframe. In fact, if we had a bond with two final cashflows in that horizon, we could interpolate the coupon cf
    # and bootstrap the maturity cf.
    max_ttm = coupon_zeros['Ttm'].max()
    mask = cashflow_df['Ttm'] <= max_ttm
    cashflow_df.loc[mask, 'Spot'] = regSpline(cashflow_df.loc[mask, 'Ttm'].values)
    cashflow_df['df'] = (1 / (1 + cashflow_df['Spot'])) ** cashflow_df['Ttm']

    # We now have a base of spot rates. Next, for every cashflow that is comes from a bond with more than one cashflow
    # outside the interpolated area, we interpolate and find its spot rate. For every bond that only has one cashflow,
    # we bootstrap.

    missing_spots = cashflow_df[cashflow_df['Spot'].isna()]

    for isin in missing_spots['ISIN']:
        bond_cf = cashflow_df[cashflow_df['ISIN'] == isin]
        # If the bond has only one cashflow left, we bootstrap


        cf_id = bond_cf[bond_cf['Spot'].isna()].index[0]
        if cf_id == 425:
            breakpoint()
        if (bond_cf['Spot'].isna().sum() == 1) and (len(bond_cf) > 1):
            spot, df = bootstrap_spot(isin, bond_cf, cashflow_df, regSpline)
            cashflow_df.loc[cf_id, 'Spot'] = spot
            cashflow_df.loc[cf_id, 'df'] = df
        else:
            # Take that cashflow from cashflow_df and interpolate to get the spot
            cashflow_df.loc[cf_id, 'Spot'] = regSpline(cashflow_df.loc[cf_id, 'Ttm']).round(6)
            cashflow_df.loc[cf_id, 'df'] = (1 / (1 + cashflow_df.loc[cf_id, 'Spot']) ** cashflow_df.loc[cf_id, 'Ttm'])
        # We rerun the spline in order to take into account the latest spots
        df_sub = cashflow_df[cashflow_df['Spot'].notna()]
        regSpline = inter.UnivariateSpline(df_sub['Ttm'], df_sub['Spot'], s=0.1)

    breakpoint()



def bootstrap_spot(isin, bond_cf, regspline):
        final_cf = bond_cf.iloc[-1]
        remaining_cfs = bond_cf.iloc[:-1]
        maturity = final_cf['Ttm']
        dirty_price = final_cf['Dirty Price']
        final_coupon = float(final_cf['Coupon'])

        sumDiscCFs = (remaining_cfs['Coupon'] * remaining_cfs['df']).sum()

        final_df = (dirty_price - sumDiscCFs) / final_coupon
        final_spot = (1 / final_df) ** (1 / maturity) - 1

        return final_spot, final_df

def main():


    bondData = pd.read_excel("BTP_data.xlsx", parse_dates=True, sheet_name=['Monitor'])
    bondData = clean_data(bondData)

    bondDatabase = bondData.copy()

    bondData = bond_cleaner(bondDatabase)

    cashflow_list = bondData.apply(lambda row: couponCalculator(row,bondData), axis=1)
    cashflows_df = pd.concat(cashflow_list.tolist(), ignore_index=True)


    curve_builder(bondData,cashflows_df)





main()