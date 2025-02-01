import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import date, timedelta
from scipy.optimize import root_scalar
from cleaning import clean_data, bond_cleaner
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

    def compute_discount_factors(self, bond_cashflows, compounding):
        """
        Compute discount factors from spot rates.

        Parameters:
            bond_cashflows (DataFrame): Dataframe containing bond cash flows.
            compounding (str): "Cont" for continuous compounding, "Discrete" otherwise.

        Returns:
            bond_cashflows (DataFrame): Updated DataFrame with calculated "df".
        """

        bond_cashflows.reset_index(drop=True, inplace=True)
        # Ensure 'df' column exists
        if 'df' not in bond_cashflows.columns:
            bond_cashflows['df'] = np.nan

        # Compute discount factors
        if compounding == "Cont":
            bond_cashflows['df'] = np.exp(-bond_cashflows['Spot'] * bond_cashflows['Ttm'])
        elif compounding == "Discrete":
            bond_cashflows['df'] = (1 / (1 + bond_cashflows['Spot'])) ** bond_cashflows['Ttm']
        else:
            raise ValueError("Invalid compounding method. Choose 'Discrete' or 'Cont'.")

        return bond_cashflows

    def compute_spot_rates(self, bond_cashflows, compounding):
        """
        Compute spot rates recursively using extracted forward rates.
        Since the forward rates we have are instantaneous, we cannot directly
        apply the discrete formula for spot rates. Instead, we need to integrate
        the forward rate curve to obtain the continuous spot rate.

        For continuous compounding, the spot rate  S(T)  is given by:
        S(T) = \frac{1}{T} \int_0^T f(t) dt
        This means we take the cumulative integral of the instantaneous forward curve
        up to  T , and then divide by  T  to get an average rate. In discrete terms,
        we approximate the integral as a summation:
        S(T) \approx \frac{1}{T} \sum_{i=0}^{T} f(t_i) \Delta t

        Parameters:
            bond_cashflows (DataFrame): Dataframe containing bond cash flows.
            compounding (str): "Cont" for continuous compounding, "Discrete" otherwise.

        Returns:
            bond_cashflows (DataFrame): Updated DataFrame with calculated "Spot".
        """

        if 'Spot' not in bond_cashflows.columns:
            bond_cashflows['Spot'] = np.nan

            # **Compute Delta_t (time intervals)**
        bond_cashflows['Delta_t'] = bond_cashflows['Ttm'].diff().fillna(bond_cashflows['Ttm'].iloc[0])

        # **Cumulative integral of forward rates**
        cumulative_integral = np.cumsum(bond_cashflows['f'] * bond_cashflows['Delta_t'])

        # **Compute spot rate for each maturity T**
        bond_cashflows['Spot'] = cumulative_integral / bond_cashflows['Ttm']

        return bond_cashflows

    def couponCalculator(self, bond, bondData, today):
        """
        Calculate the coupon payment details for a given bond and bond data.
        """
        maturity = pd.to_datetime(bond["Maturity Date"], dayfirst=True)
        issue_date = pd.to_datetime(bond["Issue Date"], dayfirst=True)
        today = pd.Timestamp.today() if today is None else today
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

    def bootstrap_spot(self, isin, bond_cf, compounding):

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

    def extract_forward_rates_for_bond(self, bond_cashflows, curve_manager):
        """
        Extracts forward rates for each time-to-maturity (Ttm) in bond_cashflows.

        Parameters:
        - bond_cashflows: DataFrame with bond cash flows including Ttm.
        - curve_manager: CurveManager instance containing the forward curve and segment boundaries.

        Returns:
        - bond_cashflows: Updated DataFrame with extracted forward rates.
        """

        f_curve = curve_manager.forward_curve
        seg_boundaries = curve_manager.segment_boundaries

        if f_curve is None or seg_boundaries is None:
            raise ValueError("Forward curve has not been initialized.")

        # Vectorized function to get forward rate
        def get_forward_rate(Ttm):
            # 🔹 If Ttm is **before** the first segment, use the first segment
            if Ttm < seg_boundaries[0]:
                segment_index = 0
            else:
                # 🔹 Find the correct segment
                for i in range(len(seg_boundaries) - 1):
                    if seg_boundaries[i] <= Ttm < seg_boundaries[i + 1]:
                        segment_index = i + 1
                        break
                else:
                    segment_index = len(seg_boundaries) - 2  # Use last segment if out of range

            # 🔹 Extract the coefficients for this segment
            a, b, c, d, e = f_curve[segment_index * 5:(segment_index + 1) * 5]

            f = a * Ttm ** 4 + b * Ttm ** 3 + c * Ttm ** 2 + d * Ttm + e
            # 🔹 Compute the forward rate f(Ttm)
            return f

        # Apply function to all Ttm values
        bond_cashflows["f"] = bond_cashflows["Ttm"].apply(get_forward_rate)

        return bond_cashflows

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

    def solve_z_spread(self, bond_cashflows, dirty_price, compounding):
        """
        Solve for the Z-spread using numerical root-finding.

        Parameters:
            bond_cashflows (DataFrame): DataFrame containing bond cash flows, spots, and discount factors.
            dirty_price (float): The observed market dirty price of the bond.
            compounding (str): "Cont" for continuous compounding, "Discrete" otherwise.

        Returns:
            float: The computed Z-spread.
        """

        def present_value_of_cashflows(x):
            """Calculate the sum of discounted cashflows given a z-spread x."""
            if compounding == "Cont":
                discounted_cf = bond_cashflows.apply(
                    lambda row: row['Coupon'] * np.exp(
                        -(row['Spot'] + x) * row['Ttm']
                    ),
                    axis=1
                )
            elif compounding == "Discrete":
                discounted_cf = bond_cashflows.apply(
                    lambda row: row['Coupon'] / (
                            (1 + row['Spot'] + x) ** row['Ttm']
                    ),
                    axis=1
                )
            else:
                raise ValueError("Invalid compounding method. Choose 'Discrete' or 'Cont'.")

            return discounted_cf.sum() - dirty_price

        # Solve for Z-spread using a numerical root-finding method
        try:
            result = root_scalar(present_value_of_cashflows, bracket=[-0.05, 0.05], method='brentq')
            return result.root if result.converged else np.nan
        except ValueError:
            return np.nan  # Assign NaN if the solver fails

    def z_spread_calculator(self, bond_cfs, zeros, final_df, compounding, curve_manager):
        """
        Calculate Z-Spreads for each bond using extracted forward rates.

        Parameters:
            bond_cfs (DataFrame): Cashflow data for all bonds.
            zeros (DataFrame): Zero-coupon bonds data.
            final_df (DataFrame): Dataframe containing optimized spot rates for maturities.
            compounding (str): Either "Cont" (Continuous) or "Discrete" for compounding method.
            curve_manager (CurveManager): Instance of CurveManager storing the forward curve.

        Returns:
            DataFrame: A dataframe with ISIN and calculated z-spreads.
        """
        z_spreads = []

        zero_isins = set(zeros['ISIN']) if not zeros.empty else set()

        # Loop through each bond in final_df
        for isin in final_df['ISIN'].unique():
            bond_des = final_df[final_df['ISIN'] == isin]['Description'].iloc[0]
            if isin in zero_isins:
                continue

            bond_cashflows = bond_cfs[bond_cfs['ISIN'] == isin].copy()
            dirty_price = bond_cashflows['Dirty Price'].iloc[0]  # The bond's actual dirty price

            bond_cashflows = self.extract_forward_rates_for_bond(bond_cashflows, curve_manager)
            bond_cashflows = self.compute_spot_rates(bond_cashflows, compounding)
            bond_cashflows = self.compute_discount_factors(bond_cashflows, compounding)

            # 🔹 **STEP 4: Solve for Z-Spread**
            z_spread = self.solve_z_spread(bond_cashflows, dirty_price, compounding)
            # check if z_spread is not nan
            if np.isnan(z_spread):
                self.logger.warning(f"Failed to calculate Z-spread for bond {bond_des}")
            else:
                z_spread = round(z_spread * 10000, 1) # Convert to basis points


            logger.info(f"Bond {bond_des}, Z-spread found: {z_spread:.1f}")

            # Store the result
            z_spreads.append({'ISIN': isin, 'Z-Spread': z_spread})

                # Convert to DataFrame
        z_spreads_df = pd.DataFrame(z_spreads)

        return z_spreads_df
