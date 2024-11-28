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
from cleaning import clean_data, bond_cleaner
from scipy.interpolate import UnivariateSpline
today = pd.to_datetime(date.today())
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None

class BondCalculator:
    def __init__(self):
        # Initialize any necessary attributes if needed
        pass

    def extract_last_forward(self, X, segment_boundaries, updated_spot_df):
        """
        Calculate the last forward rate from the given coefficients and segment boundaries.
        """
        # Take the last Ttm
        last_t = updated_spot_df['Ttm'].iloc[-1]
        # Calculate the forward rate using the polynomial coefficients
        a, b, c, d, e = X[-5:]
        f = a * last_t ** 4 + b * last_t ** 3 + c * last_t ** 2 + d * last_t + e
        return f

    def build_forward_curve_from_coefficients(self, X, segment_boundaries):
        """
        Build and plot the forward curve from coefficients, given segment boundaries.
        """
        n_segments = len(segment_boundaries)
        nodes = n_segments - 1
        coefficients = X.reshape((n_segments, 5))  # Reshape X into segments, 5 coefficients each

        # Define colors for each segment
        colours = ['blue', 'orange', 'yellow', 'green', 'red', 'black']

        # Plot each segment
        plt.figure(figsize=(10, 6))
        for i in range(nodes):
            T_start, T_end = segment_boundaries[i], segment_boundaries[i + 1]
            a, b, c, d, e = coefficients[i]

            # Generate points for the segment
            T_values = np.linspace(T_start, T_end, 100)
            f_values = a * T_values ** 4 + b * T_values ** 3 + c * T_values ** 2 + d * T_values + e

            # Plot the segment
            plt.plot(T_values, f_values, color=colours[0], label=f'Segment {i + 1} ({T_start} to {T_end})')

        # Add labels and legend
        plt.xlabel("Time to Maturity (T)")
        plt.ylabel("Forward Rate f(T)")
        plt.title("Forward Rate Curve Segmented by Maturities")
        plt.legend()
        plt.grid(True)
        plt.show()

    def couponCalculator(self, bond, bondData):
        """
        Calculate the coupon payment details for a given bond and bond data.
        """
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

        # Convert to a DataFrame for better visualization
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

        # Add final coupon payment with face value at maturity
        cashflow_df.loc[cashflow_df['CF Date'] == maturity, 'Coupon'] = float(coupon) + 100
        cashflow_df['Ttm'] = ((cashflow_df['CF Date'] - today).dt.days / 365).round(3)
        cashflow_df['Spot'] = pd.to_numeric(cashflow_df['Spot'], errors='coerce')

        # Generate a random ISIN if missing
        if cashflow_df['ISIN'].isna().any():
            new_isin = 'IT' + ''.join([str(np.random.randint(0, 10)) for _ in range(10)])
            cashflow_df['ISIN'] = new_isin
        return cashflow_df

    def bootstrap_spot(self, isin, bond_cf, regspline, compounding):

        final_cf = bond_cf.iloc[-1]
        remaining_cfs = bond_cf.iloc[:-1]
        maturity = final_cf['Ttm']
        dirty_price = final_cf['Dirty Price']
        final_coupon = float(final_cf['Coupon'])

        if compounding == 'Cont':
            sumDiscCFs = (remaining_cfs['Coupon'] * np.exp(-remaining_cfs['Spot'] * remaining_cfs['Ttm'])).sum()
            final_df = (dirty_price - sumDiscCFs) / final_coupon
            final_spot = -np.log(final_df) / maturity
        else:
            sumDiscCFs = (remaining_cfs['Coupon'] * remaining_cfs['df']).sum()
            final_df = (dirty_price - sumDiscCFs) / final_coupon
            final_spot = (1 / final_df) ** (1 / maturity) - 1

        return final_spot, final_df


    def calculate_price_from_forward_curve(self, bond_cf):
        """
        Calculates the price of a bond using the forward curve for discounting each cash flow.

        Parameters:
        - bond_cf: DataFrame with bond cash flows. Must contain:
          - 'Coupon': Actual cash flows (including final principal if applicable).
          - 'df': Discount factors for each cash flow.

        Returns:
        - price: The estimated bond price based on the discount factors.
        """
        price = (bond_cf['Coupon'] * bond_cf['df']).sum()
        return price