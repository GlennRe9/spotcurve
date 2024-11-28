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
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from cleaning import clean_data, bond_cleaner
from bond_calculations import BondCalculator
from random import randrange
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

def driver(spot_df, coupon_bonds, cashflow_df,bond_calculator, compounding):
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
    regSpline = inter.UnivariateSpline(spot_df['Ttm'], spot_df['Spot'], s=0.1)
    # Use linear interpolation instead of spline for calculating spot rates
    ind = spot_df['Ttm'].values.reshape(-1, 1)  # Independent variable (Ttm)
    dep = spot_df['Spot'].values  # Dependent variable (Spot)
    linear_model = LinearRegression()
    linear_model.fit(ind, dep)

    # Store estimated spot rates for the zero-coupon bonds
    estimated_spot_rates = list(spot_df['Spot'])

    updated_spot_df = spot_df
    # Loop through each coupon bond iteratively
    for i, coupon_bond in coupon_bonds.iterrows():
        logger.info(f"Processing coupon bond {i + 1} with maturity {coupon_bond['Ttm']} years.")

        # Step 1: Bootstrap the initial spot rate for the current coupon bond
        isin = coupon_bond['ISIN']
        des = coupon_bond['Description']
        bond_cf = cashflow_df[cashflow_df['ISIN'] == isin]
        assert(len(bond_cf) > 1, f"Invalid cash flow data for bond {i + 1} with ISIN {isin}.")
        bond_cf.sort_values(by='Ttm', inplace=True, ignore_index=True)

        # Step 2: Exclude the final cash flow and apply the spline to earlier maturities
        final_cf = bond_cf.iloc[-1]  # Last cash flow
        earlier_cfs = bond_cf.iloc[:-1]  # All cash flows except the last one
        max_ttm = earlier_cfs['Ttm'].max()
        mask = earlier_cfs['Ttm'] <= max_ttm

        # Apply the interp to get the spot rates for earlier cash flows
        earlier_cfs.loc[mask, 'Spot'] = linear_model.predict(earlier_cfs.loc[mask, 'Ttm'].values.reshape(-1, 1))
        if compounding == 'Cont':
            earlier_cfs['df'] = np.exp(-earlier_cfs['Spot'] * earlier_cfs['Ttm'])
            bond_cf['df'] = np.exp(-bond_cf['Spot'] * bond_cf['Ttm'])
        else:
            earlier_cfs['df'] = (1 / (1 + earlier_cfs['Spot'])) ** earlier_cfs['Ttm']
            bond_cf['df'] = (1 / (1 + bond_cf['Spot'])) ** bond_cf['Ttm']

        # Bootstrapping
        initial_spot_rate, initial_df = bond_calculator.bootstrap_spot(isin, bond_cf, regSpline, compounding)

        logger.info(f"Initial spot rate for coupon bond {i + 1}: {initial_spot_rate:.6f}")
        # Update the 'Spot' for the last cash flow (final_cf) after bootstrapping
        bond_cf.loc[bond_cf.index[-1], 'Spot'] = initial_spot_rate
        bond_cf.loc[bond_cf.index[-1], 'df'] = initial_df

        final_cf = bond_cf.iloc[[-1]]  # Last cash flow
        final_cf.drop(columns=['CF Date'], inplace=True)

        # Step 2: Append this new spot rate to coupon_zeros
        updated_spot_df = pd.concat([updated_spot_df, final_cf]).sort_values(by='Ttm').reset_index(drop=True)
        updated_spot_df.loc[updated_spot_df.index[-1], 'Delta_t'] = updated_spot_df['Ttm'].iloc[-1] - \
                                                                    updated_spot_df['Ttm'].iloc[-2]
        x = optimisation.run_optimisation(updated_spot_df, compounding)
        segment_boundaries = updated_spot_df['Ttm'].values

        #f = bond_calculator.extract_last_forward(x, segment_boundaries,updated_spot_df)
        # Loop through each Ttm in updated_spot_df to update forward rates at each node
        optimised_spot_df = updated_spot_df.copy()
        for i, ttm in enumerate(segment_boundaries):
            # Identify the appropriate segment for this Ttm
            segment_index = min(i, len(segment_boundaries) - 1)  # Avoids going out of bounds
            a, b, c, d, e = x[segment_index * 5:(segment_index + 1) * 5]

            # Calculate forward rate for this specific Ttm using the segment's polynomial coefficients
            # I believe this one is assigned to the forward rate from 0 to 0+n, because the first
            #constraint calculates the coefficients from the left of the first node to the right of the first node
            forward_rate = a * ttm ** 4 + b * ttm ** 3 + c * ttm ** 2 + d * ttm + e

            #ttm2 = updated_spot_df['Ttm'].iloc[i + 1]
            # Update the forward rate in updated_spot_df at this Ttm
            optimised_spot_df.loc[optimised_spot_df['Ttm'] == ttm, 'f'] = forward_rate

            logger.info(f"Updated forward rate at Ttm={ttm} is {forward_rate:.6f}")

        # We start from the last bond, working our way down from forward to discount factors
        for i in range(len(optimised_spot_df) - 1, 0, -1):
            if compounding == "Cont":
                # Continuous method
                optimised_spot_df.loc[i, 'df'] = optimised_spot_df.loc[i + 1, 'df'] * np.exp(
                    -optimised_spot_df.loc[i + 1, 'f'] * optimised_spot_df.loc[i + 1, 'Delta_t']
                )
            elif compounding == "Discrete":
                # Discrete method
                new_df = optimised_spot_df.loc[i -1, 'df'] / (1 + optimised_spot_df.loc[i, 'f']) ** optimised_spot_df.loc[i, 'Delta_t']
                optimised_spot_df.loc[i, 'df'] = new_df
                logger.info(f" Old discount factor: { optimised_spot_df.loc[i, 'df']}, New discount factor: {new_df}")
            else:
                raise ValueError("Invalid compounding method. Choose 'Discrete' or 'Cont'.")

        # Recalculate the spot rates from the updated discount factors
        if compounding == "Cont":
            optimised_spot_df['Spot'] = -np.log(optimised_spot_df['df']) / optimised_spot_df['Ttm']
        elif compounding == "Discrete":
            optimised_spot_df['Spot'] = (1 / optimised_spot_df['df']) ** (1 / optimised_spot_df['Ttm']) - 1

        # # We update with the PQPI method
        # updated_spot_df.loc[updated_spot_df['ISIN'] == isin, 'f'] = f

        updated_spot_df = perturbation_function(optimised_spot_df,bond_calculator, bond_cf)

        ind = updated_spot_df['Ttm'].values.reshape(-1, 1)  # Independent variable (Ttm)
        dep = updated_spot_df['Spot'].values  # Dependent variable (Spot)
        linear_model = LinearRegression()
        linear_model.fit(ind, dep)



    return x, segment_boundaries, updated_spot_df
        #breakpoint()
        #build_forward_curve_from_coefficients(x, segment_boundaries)
        #estimated_px =

        #bond_calculator.build_forward_curve_from_coefficients(x, segment_boundaries)
        # breakpoint()
        # # # Call this after optimisation to evaluate or plot the forward curve
        # # optimized_X = optimisation.run_optimisation(coupon_zeros)
        # # forward_curve = build_forward_curve_from_coefficients(optimized_X, segment_boundaries)
        #
        # breakpoint()


# Function to create and evaluate the forward curve
def build_forward_curve_from_coefficients(X, segment_boundaries):
    n_segments = len(segment_boundaries) + 1 # Adding the n+1 seg
    nodes = n_segments - 1
    coefficients = X.reshape((n_segments, 5))  # Reshape x into 3 segments, 5 coefficients each

    # Define colors for each segment
    colors = ['blue', 'orange', 'yellow', 'green', 'red', 'black', 'purple', 'brown', 'pink', 'gray']

    # Add segment from 0
    segment_boundaries = np.insert(segment_boundaries, 0, 0)
    # Add a final node at the end of size 1
    segment_boundaries = np.append(segment_boundaries, segment_boundaries[-1] + 1)

    # Plot each segment
    plt.figure(figsize=(10, 6))
    for i in range(nodes):  # We only plot the first two segments
        # random number between 0 and 7
        col = colors[randrange(7)]
        T_start, T_end = segment_boundaries[i], segment_boundaries[i + 1]
        a, b, c, d, e = coefficients[i]

        # Generate points for the segment
        T_values = np.linspace(T_start, T_end, 100)
        f_values = a * T_values ** 4 + b * T_values ** 3 + c * T_values ** 2 + d * T_values + e

        # Plot the segment
        plt.plot(T_values, f_values, color=col, label=f'Segment {i + 1} ({T_start} to {T_end})')

    # Add labels and legend
    plt.xlabel("Time to Maturity (T)")
    plt.ylabel("Forward Rate f(T)")
    plt.title("Forward Rate Curve Segmented by Maturities")
    plt.legend()
    plt.grid(True)
    plt.show()

def perturbation_function(updated_spot_df,bond_calculator, bond_cf):
    s = updated_spot_df['Spot'].iloc[-1]
    f = updated_spot_df['f'].iloc[-1]
    df = updated_spot_df['df'].iloc[-1]
    delta_t = updated_spot_df['Delta_t'].iloc[-1]
    Ttm = updated_spot_df['Ttm'].iloc[-1]
    isin = updated_spot_df['ISIN'].iloc[-1]
    act_price = updated_spot_df['Dirty Price'].iloc[-1] / 100
    est_price = bond_calculator.calculate_price_from_forward_curve(bond_cf) / 100


    it_count = 0
    max_iterations = 100

    while True:
        # Perturbation
        s_u = s + s / 100
        s_d = s - s / 100
        df_u = (1 / (1 + s_u)) ** Ttm
        df_d = (1 / (1 + s_d)) ** Ttm
        f_u = ( updated_spot_df['df'].iloc[-2] / df_u ) ** ( 1 / delta_t ) - 1
        f_d = ( updated_spot_df['df'].iloc[-2] / df_d ) ** ( 1 / delta_t ) - 1
        p_u = df_u
        p_d = df_d

        # f_u = df_u / updated_spot_df['df'].iloc[-2]
        # f_d = df_d / updated_spot_df['df'].iloc[-2]
        # p_u = bond_calculator.calculate_price_from_forward_curve(df_u, bond_cf)
        # p_d = bond_calculator.calculate_price_from_forward_curve(df_d, bond_cf)

        p_deriv = (p_u - p_d) / (s / 50)
        y_deriv = (s_u - s_d) / (df / 50)
        new_s = s + y_deriv * (act_price - est_price)
        new_df = (1 / (1 + new_s)) ** Ttm
        new_f = ( updated_spot_df['df'].iloc[-2] / new_df ) ** ( 1 / delta_t ) - 1
        logger.info(f'New forward found at {new_f}')

        updated_spot_df.loc[updated_spot_df.index[-1], 'Spot'] = new_s
        updated_spot_df.loc[updated_spot_df.index[-1], 'df'] = new_df
        updated_spot_df.loc[updated_spot_df.index[-1], 'f'] = new_f

        # Calculate spot difference for convergence
        spot_diff = abs(s - new_s)

        # Break if the condition is met
        if spot_diff <= 1e-2:
            logger.info(f"Convergence achieved with spot difference: {spot_diff}")
            break

        # Break if max iterations reached
        if it_count >= max_iterations:
            logger.warning(f"Max iterations reached without convergence. Final spot difference: {spot_diff}")
            break

        # # Update the forward curve using PQPI optimisation after each perturbation
        # x = optimisation.run_optimisation(updated_spot_df)
        # segment_boundaries = updated_spot_df['Ttm'].values
        # f = bond_calculator.extract_last_forward(x, segment_boundaries, updated_spot_df)

        # Log the updated forward rate and update the DataFrame
        logger.info(f"forward rate being updated to {f}")
        updated_spot_df.loc[updated_spot_df['ISIN'] == isin, 'f'] = f

        # Update for the next iteration
        s = new_s
        est_price = new_df
        df = new_df
        it_count += 1  # Increment iteration counter


    return updated_spot_df

def curve_builder(bondDatabase, cashflow_df,bond_calculator, compounding):

    # Pure zeros are actual zero-coupon bonds
    # Non-Pure zeroes are coupon bonds which only have one cashflow left
    pure_zeros = bondDatabase[(bondDatabase['Coupon'] == 0) & (bondDatabase['Next Coupon Date'].isna())]
    logging.info(f'Number of zero-coupon bonds: {len(pure_zeros)}')
    coupon_zeros = bondDatabase[(pd.isna(bondDatabase['Next Coupon Date'])) & (bondDatabase['Coupon'] != 0)]
    logger.info(f'Number of coupon zeros: {len(coupon_zeros)}')

    zeros = pd.concat([pure_zeros, coupon_zeros], axis=0)
    zeros['Spot'] = zeros['Yield to Maturity']
    zeros.sort_values(by='Ttm', inplace=True)
    zeros['Spot'] = pd.to_numeric(zeros['Spot'], errors='coerce')
    zeros['Ttm'] = pd.to_numeric(zeros['Ttm'], errors='coerce')

    zeros['Delta_t'] = zeros['Ttm'].diff()
    zeros['Delta_t'].iloc[0] = zeros['Ttm'].iloc[0]

    if compounding == 'Cont':
        zeros['df'] = np.exp(-zeros['Spot'] * zeros['Ttm'])  # Using continuous compounding formula
        # Calculate the instantaneous forward rate
        delta_t = zeros['Ttm'].diff().fillna(zeros['Ttm'].iloc[0])  # Ensure no NaN for the first delta
        zeros['Spot_shifted_up'] = zeros['Spot'].shift(-1)
        zeros['Spot_shifted_down'] = zeros['Spot'].shift(1)
        # Central difference method for derivative of spot rates
        zeros['Spot_derivative'] = (zeros['Spot_shifted_up'] - zeros['Spot_shifted_down']) / (2 * delta_t)
        zeros['Spot_derivative'].iloc[0] = (zeros['Spot'].iloc[1] - zeros['Spot'].iloc[0]) / delta_t.iloc[1]
        zeros['Spot_derivative'].iloc[-1] = (zeros['Spot'].iloc[-1] - zeros['Spot'].iloc[-2]) / delta_t.iloc[-1]
        zeros['f'] = zeros['Spot'] + zeros['Ttm'] * zeros['Spot_derivative']
        zeros.drop(['Spot_shifted_up', 'Spot_shifted_down', 'Spot_derivative'], axis=1, inplace=True)
    else:
        zeros['df'] = (1 / (1 + zeros['Spot'])) ** zeros['Ttm']
        # divide zero['df'] by its previous value to get the forward rate
        zeros['f'] = (zeros['df'].shift(1) / zeros['df']) ** (1 / zeros['Delta_t']) - 1



    # coupon bonds is the the bondDatabase where the isin is not in zeros
    coupon_bonds = bondDatabase[~bondDatabase['ISIN'].isin(zeros['ISIN'])].reset_index(drop=True)
    assert(len(zeros) + len(coupon_bonds) == len(bondDatabase))

    x, segment_boundaries, final_df = driver(zeros, coupon_bonds, cashflow_df,bond_calculator,compounding)
    build_forward_curve_from_coefficients(x, segment_boundaries)
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

def main():

    bond_calculator = BondCalculator()

    sheet = 'TestCurve' # Monitor #hardcopy # TestCurve

    bondData = pd.read_excel("BTP_data.xlsx", parse_dates=True, sheet_name=[sheet])
    bondData = clean_data(bondData, sheet)

    bondDatabase = bondData.copy()

    bondData = bond_cleaner(bondDatabase)

    cashflow_list = bondData.apply(lambda row: bond_calculator.couponCalculator(row,bondData), axis=1)
    cashflows_df = pd.concat(cashflow_list.tolist(), ignore_index=True)

    compounding = 'Discrete'

    curve_builder(bondData,cashflows_df,bond_calculator, compounding)

main()