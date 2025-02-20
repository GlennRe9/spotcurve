import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import date, timedelta
import optimisation as optimisation
from sklearn.linear_model import LinearRegression

today = pd.to_datetime(date.today())
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None


class CurveManager:
    def __init__(self):
        self.forward_curve = None  # Store forward curve coefficients
        self.segment_boundaries = None  # Store segment boundaries
        self.compounding = "Discrete"

    def store_forward_curve(self, x, segment_boundaries):
        """Store the forward curve for later use."""
        self.forward_curve = x
        self.segment_boundaries = segment_boundaries

    def perturbation_function(self, updated_spot_df, bond_calculator, bond_cf):
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
            f_u = (updated_spot_df['df'].iloc[-2] / df_u) ** (1 / delta_t) - 1
            f_d = (updated_spot_df['df'].iloc[-2] / df_d) ** (1 / delta_t) - 1
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
            new_f = (updated_spot_df['df'].iloc[-2] / new_df) ** (1 / delta_t) - 1
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
    def driver(self, spot_df, coupon_bonds, cashflow_df, bond_calculator, compounding):
        """
        Handles the full iteration process, including bootstrapping, perturbation, and forward curve optimization.
        """
        # Start with zero-coupon bonds (cf_data is a copy of coupon_zeros)
        cf_data = spot_df.copy()

        # We use regspline here to calculate the cashflow yields
        # regSpline = inter.UnivariateSpline(spot_df['Ttm'], spot_df['Spot'], s=0.1)
        # Use linear interpolation instead of spline for calculating spot rates
        ind = spot_df['Ttm'].values.reshape(-1, 1)  # Independent variable (Ttm)
        dep = spot_df['Spot'].values  # Dependent variable (Spot)
        linear_model = LinearRegression()
        linear_model.fit(ind, dep)

        # Store estimated spot rates for the zero-coupon bonds
        estimated_spot_rates = list(spot_df['Spot'])

        updated_spot_df = spot_df
        bond_cfs = pd.DataFrame()

        # Loop through each coupon bond iteratively
        for i, coupon_bond in coupon_bonds.iterrows():
            logger.info(f"Processing coupon bond {i + 1} with maturity {coupon_bond['Ttm']} years.")

            # Step 1: Bootstrap the initial spot rate for the current coupon bond
            isin = coupon_bond['ISIN']
            des = coupon_bond['Description']
            bond_cf = cashflow_df[cashflow_df['ISIN'] == isin]
            assert (len(bond_cf) > 1, f"Invalid cash flow data for bond {i + 1} with ISIN {isin}.")
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

            # Add cashflows with spot rates to all_cashflows_with_spots
            bond_cfs = pd.concat([bond_cfs, earlier_cfs], ignore_index=True)

            # Bootstrapping
            initial_spot_rate, initial_df = bond_calculator.bootstrap_spot(isin, bond_cf, compounding)
            logger.info(f"Initial spot rate for coupon bond {i + 1}: {initial_spot_rate:.6f}")

            # Update the 'Spot' for the last cash flow (final_cf) after bootstrapping
            bond_cf.loc[bond_cf.index[-1], 'Spot'] = initial_spot_rate
            bond_cf.loc[bond_cf.index[-1], 'df'] = initial_df

            # # We store the cash flows in the bond_cfs dataframe
            # bond_cfs = pd.concat([bond_cfs, bond_cf.copy()], ignore_index=True)

            final_cf = bond_cf.iloc[[-1]]  # Last cash flow
            final_cf.drop(columns=['CF Date'], inplace=True)

            # Step 2: Append this new spot rate to coupon_zeros
            updated_spot_df = pd.concat([updated_spot_df, final_cf]).sort_values(by='Ttm').reset_index(drop=True)
            updated_spot_df.loc[updated_spot_df.index[-1], 'Delta_t'] = updated_spot_df['Ttm'].iloc[-1] - \
                                                                        updated_spot_df['Ttm'].iloc[-2]
            x = optimisation.run_optimisation(updated_spot_df, compounding)
            segment_boundaries = updated_spot_df['Ttm'].values

            # Loop through each Ttm in updated_spot_df to update forward rates at each node
            optimised_spot_df = updated_spot_df.copy()
            for i, ttm in enumerate(segment_boundaries):
                # Identify the appropriate segment for this Ttm
                segment_index = min(i, len(segment_boundaries) - 1)  # Avoids going out of bounds
                a, b, c, d, e = x[segment_index * 5:(segment_index + 1) * 5]

                # Calculate forward rate for this specific Ttm using the segment's polynomial coefficients
                # I believe this one is assigned to the forward rate from 0 to 0+n, because the first
                # constraint calculates the coefficients from the left of the first node to the right of the first node
                forward_rate = a * ttm ** 4 + b * ttm ** 3 + c * ttm ** 2 + d * ttm + e

                # ttm2 = updated_spot_df['Ttm'].iloc[i + 1]
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
                    new_df = optimised_spot_df.loc[i - 1, 'df'] / (1 + optimised_spot_df.loc[i, 'f']) ** \
                             optimised_spot_df.loc[i, 'Delta_t']
                    optimised_spot_df.loc[i, 'df'] = new_df
                    logger.info(
                        f" Old discount factor: {optimised_spot_df.loc[i, 'df']}, New discount factor: {new_df}")
                else:
                    raise ValueError("Invalid compounding method. Choose 'Discrete' or 'Cont'.")

            # Recalculate the spot rates from the updated discount factors
            if compounding == "Cont":
                optimised_spot_df['Spot'] = -np.log(optimised_spot_df['df']) / optimised_spot_df['Ttm']
            elif compounding == "Discrete":
                optimised_spot_df['Spot'] = (1 / optimised_spot_df['df']) ** (1 / optimised_spot_df['Ttm']) - 1

            # # We update with the PQPI method
            # updated_spot_df.loc[updated_spot_df['ISIN'] == isin, 'f'] = f

            updated_spot_df = self.perturbation_function(optimised_spot_df, bond_calculator, bond_cf)


            # Rerunning optimisation
            x = optimisation.run_optimisation(updated_spot_df, compounding)
            updated_spot_df = optimisation.prep_optimisation(updated_spot_df, x, segment_boundaries, compounding)

            ind = updated_spot_df['Ttm'].values.reshape(-1, 1)  # Independent variable (Ttm)
            dep = updated_spot_df['Spot'].values  # Dependent variable (Spot)
            linear_model = LinearRegression()
            linear_model.fit(ind, dep)
            # Store the forward curve in the CurveManager class
            self.store_forward_curve(x, segment_boundaries)

        return x, segment_boundaries, updated_spot_df, bond_cfs