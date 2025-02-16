import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging, os
from datetime import date
import json
import statistics
import re
import constraints as smoothing
import optimisation as optimisation
from datetime import date, timedelta
import datetime
import scipy.interpolate as inter
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from cleaning import clean_data, bond_cleaner, find_on_the_run
from bond_calculations import BondCalculator
from curve_manager import CurveManager
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

# Useful resources: https://quant.stackexchange.com/questions/3302/deriving-spot-rates-from-treasury-yield-curve

def curve_visualiser(spline, x, y, x1, x2):
    myX = np.linspace(x1, x2, 100)
    plt.plot(myX, spline(myX), label='firsthalf')
    plt.axis([x1, x2, -0.01, 0.04])
    plt.scatter(x, y)
    plt.show()


# Function to create and evaluate the forward curve
def build_forward_curve_from_coefficients(X, segment_boundaries):
    n_segments = len(segment_boundaries) + 1  # Adding the n+1 seg
    nodes = n_segments - 1
    coefficients = X.reshape((n_segments, 5))  # Reshape x into segments, 5 coefficients each

    # Define colors for each segment
    colors = ['blue', 'orange', 'yellow', 'green', 'red', 'black', 'purple', 'brown', 'pink', 'gray']

    # Add segment from 0
    segment_boundaries = np.insert(segment_boundaries, 0, 0)
    # Add a final node at the end of size 1
    segment_boundaries = np.append(segment_boundaries, segment_boundaries[-1] + 1)

    # Plot each segment
    plt.figure(figsize=(10, 6))
    for i in range(nodes):
        col = colors[randrange(len(colors))]
        T_start, T_end = segment_boundaries[i], segment_boundaries[i + 1]
        a, b, c, d, e = coefficients[i]

        # Generate points for the segment
        T_values = np.linspace(T_start, T_end, 100)
        f_values = a * T_values ** 4 + b * T_values ** 3 + c * T_values ** 2 + d * T_values + e

        # Plot the segment
        plt.plot(T_values, f_values, color=col, label=f'Segment {i + 1} ({T_start:.3f} to {T_end:.3f})')

    # Add labels and legend
    plt.xlabel("Time to Maturity (T)")
    plt.ylabel("Forward Rate f(T)")
    plt.title("Forward Rate Curve Segmented by Maturities")

    # Adjust the legend position to avoid overlap
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.grid(True)

    # Tighten layout to make space for the legend
    plt.tight_layout()

    plt.show()

def build_forward_and_spot_curve(x, segment_boundaries):
    n_segments = len(segment_boundaries) + 1  # Adding the n+1 segment
    nodes = n_segments - 1
    coefficients = x.reshape((n_segments, 5))  # Reshape x into segments, 5 coefficients each

    # Define colors for each segment
    colors = ['blue', 'orange', 'yellow', 'green', 'red', 'black', 'purple', 'brown', 'pink', 'gray']
    spot_color = '#FFD700'  # Bright shade of gold

    # Add segment from 0
    segment_boundaries = np.insert(segment_boundaries, 0, 0)
    segment_boundaries = np.append(segment_boundaries, segment_boundaries[-1] + 1)

    # Arrays to store the full curve (forward rates and T_values)
    all_T_values = []
    all_f_values = []

    # Generate forward rates and T_values for all segments
    for i in range(nodes):
        col = colors[i % len(colors)]
        T_start, T_end = segment_boundaries[i], segment_boundaries[i + 1]
        a, b, c, d, e = coefficients[i]

        # Generate points for the segment
        T_values = np.linspace(T_start, T_end, 100)
        f_values = a * T_values ** 4 + b * T_values ** 3 + c * T_values ** 2 + d * T_values + e

        # Store the segment values in the full curve arrays
        all_T_values.append(T_values)
        all_f_values.append(f_values)

        # Plot the forward curve for this segment
        plt.plot(T_values, f_values, color=col, label=f'Forward Curve {i + 1} ({T_start:.3f} to {T_end:.3f})')

    # Concatenate all segments into continuous arrays
    all_T_values = np.concatenate(all_T_values)
    all_f_values = np.concatenate(all_f_values)

    # Remove the first T_value if it is zero (to avoid division by zero in spot calculation)
    valid_indices = all_T_values > 0
    all_T_values = all_T_values[valid_indices]
    all_f_values = all_f_values[valid_indices]

    # Compute the cumulative integral of forward rates over all segments
    cumulative_integral = np.cumsum(all_f_values * np.gradient(all_T_values))  # Approximate the integral

    # Compute the spot rates
    spot_rates = cumulative_integral / all_T_values

    # Plot the spot curve (continuous line)
    plt.plot(all_T_values, spot_rates, color=spot_color, linewidth=2, label="Spot Curve")

    # Add labels and legend
    plt.xlabel("Time to Maturity (T)")
    plt.ylabel("Rate (%)")
    plt.title("Forward Curve & Spot Curve - BTP curve - 03-02-2025")

    # Adjust the legend position to avoid overlap
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.grid(True)

    # Tighten layout to make space for the legend
    plt.tight_layout()

    plt.show()

def curve_builder(bondDatabase, zeros, cashflow_df, bond_calculator, curve_manager, compounding):
    if zeros is None:
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
    # assert(len(zeros) + len(coupon_bonds) == len(bondDatabase))

    x, segment_boundaries, final_df, bond_cfs = curve_manager.driver(zeros, coupon_bonds, cashflow_df, bond_calculator,
                                                                     compounding)

    # Save the forward curve in CurveManager
    curve_manager.store_forward_curve(x, segment_boundaries)

    # Visualising the forward curve
    # build_forward_curve_from_coefficients(x, segment_boundaries)
    build_forward_and_spot_curve(x, segment_boundaries)

    # Z spread calculation
    z_spreads = bond_calculator.z_spread_calculator(cashflow_df, zeros, final_df, compounding, curve_manager)

    # Merge final_df with z_spreads on the ISIN column
    final_df = final_df.merge(z_spreads, on="ISIN", how="left")
    logger.info(f"Updated final_df with Z-Spreads:\n{final_df}")

    # optimisation.iterative_optimisation(zeros, coupon_bonds)

    # result = optimisation.run_optimisation(coupon_zeros)


def main():
    bond_calculator = BondCalculator()
    curve_manager = CurveManager()
    # today = pd.Timestamp(datetime.date(2012, 2, 10))

    sheet = 'hardcopy'  # Monitor #hardcopy # TestCurve # USCurve

    # Ensure the path correctly points to the spotcurve directory
    spotcurve_path = os.path.dirname(os.path.abspath(__file__))  # Gets the path of main.py
    btp_file_path = os.path.join(spotcurve_path, "BTP_data.xlsx")
    bondData = pd.read_excel(btp_file_path, parse_dates=True, sheet_name=[sheet])
    bondData = clean_data(bondData, sheet, today)

    bondDatabase = bondData.copy()

    bondData = bond_cleaner(bondDatabase)
    bondData, zeros = find_on_the_run(bondData)

    cashflow_list = bondData.apply(lambda row: bond_calculator.couponCalculator(row, bondData, today), axis=1)
    cashflows_df = pd.concat(cashflow_list.tolist(), ignore_index=True).sort_values(by=['CF Date'])

    compounding = 'Discrete'

    curve_builder(bondData, zeros, cashflows_df, bond_calculator, curve_manager, compounding)

    return curve_manager.forward_curve, curve_manager.segment_boundaries

if __name__ == "__main__":
    main()