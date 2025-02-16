import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import block_diag, solve
import constraints as constraints
import logging
logger = logging.getLogger(__name__)


import numpy as np
from scipy.optimize import minimize, LinearConstraint

def objective_function(X, H):
    """
    Quadratic objective function: X' H X.
    This minimizes the integral of the squared second derivative of the forward curve.

    Parameters:
    X (numpy array): Coefficient vector for the quartic polynomials.
    H (numpy array): Matrix associated with the smoothness penalty (based on second derivatives).

    Returns:
    float: The value of the objective function (smoothness penalty).
    """
    return 0.5 * np.dot(X.T, np.dot(H, X))

def prep_optimisation(updated_spot_df, x, segment_boundaries, compounding):
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

    return optimised_spot_df


def run_optimisation(clean_data, compounding):
    # Number of bonds
    m = len(clean_data)

    # Build the constraints
    A, B = constraints.build_pricing_constraints(clean_data, compounding)

    # Define the segment boundaries based on the 'Ttm' column
    segment_boundaries = clean_data['Ttm'].values  # Assumes 'Ttm' contains the time-to-maturity for each bond
    # Adding the 0 point
    segment_boundaries = np.insert(segment_boundaries, 0, 0.0)
    # Construct the smoothness penalty matrix H based on the specified structure
    H = construct_smoothness_penalty_matrix(segment_boundaries)

    # Run the KKT-based optimisation
    X, lambda_values = run_optimisation_via_kkt(A, B, H)

    return X





def construct_smoothness_penalty_matrix(segment_boundaries):
    """
    Construct the smoothness penalty matrix H based on the second derivatives of
    the quartic polynomials for each segment, using the specified structure in the paper.

    Parameters:
    segment_boundaries (list of float): The boundaries [T_0, T_1, ..., T_m+1] for each segment.

    Returns:
    H (numpy array): Smoothness penalty matrix for the objective function.
    """
    m = len(segment_boundaries)  # Number of segments
    n_coefficients = 5 * (m)  # Total number of coefficients (5 per segment)
    H = np.zeros((n_coefficients, n_coefficients))  # Initialize H matrix

    # Loop through each segment and calculate the block for H based on the second derivative
    for i in range(m-1):
        T_start, T_end = segment_boundaries[i], segment_boundaries[i + 1]
        delta_i = T_end - T_start  # Interval length

        # Define the entries for the h_i matrix
        h_matrix = np.array([
            [144 / 5 * delta_i ** 5, 18 * delta_i ** 4, 8 * delta_i ** 3, 0, 0],
            [18 * delta_i ** 4, 12 * delta_i ** 3, 6 * delta_i ** 2, 0, 0],
            [8 * delta_i ** 3, 6 * delta_i ** 2, 4 * delta_i, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        # Place h_matrix in the block-diagonal position within H
        start_idx = i * 5
        end_idx = start_idx + 5
        H[start_idx:end_idx, start_idx:end_idx] = h_matrix

    return H

def run_optimisation_via_kkt(A, B, H):
    """
    Solve the optimization problem using the KKT system.

    Parameters:
    A (numpy array): Constraint matrix.
    B (numpy array): Target values for constraints.
    H (numpy array): Smoothness penalty matrix for the objective function.

    Returns:
    X (numpy array): Solution vector containing the coefficients for the forward curve.
    """
    # Dimensions
    num_constraints = A.shape[0]
    num_coefficients = H.shape[0]

    # Build the KKT matrix
    KKT_matrix = np.block([
        [2 * H, A.T],
        [A, np.zeros((num_constraints, num_constraints))]
    ])

    # Build the right-hand side vector
    rhs_vector = np.concatenate([np.zeros(num_coefficients), B])

    # Solve the KKT system
    solution = np.linalg.solve(KKT_matrix, rhs_vector)

    # Extract X (polynomial coefficients) and lambda (Lagrange multipliers)
    X = solution[:num_coefficients]
    lambda_values = solution[num_coefficients:]

    return X, lambda_values

def iterative_optimisation(coupon_zeros, coupon_bonds):
    """
    Iteratively construct a smooth forward curve by adding bonds one-by-one.
    """
    # Initial optimization using zero-coupon bonds
    cf_data = coupon_zeros.copy()

    estimated_spot_rates = list(coupon_zeros['Spot'])
    first_coupon_bond = coupon_bonds.iloc[0]
    logger.info(f"Processing the first coupon bond with maturity {first_coupon_bond['Ttm']} years.")

    initial_spot_rate = bootstrap_spot(cf_data, first_coupon_bond)
    logger.info(f"Initial spot rate for the first coupon bond: {initial_spot_rate:.6f}")




    # Store the estimated spot rates for each bond
    estimated_spot_rates = list(coupon_zeros['Spot'])

    # Loop through each new coupon bond and iteratively add it to the optimization
    for i in range(len(coupon_bonds)):
        print(f"Adding coupon bond {i + 1} to the optimization...")

        # Estimate initial spot rate for the new bond
        new_bond = coupon_bonds.iloc[i]
        T_i = new_bond['Ttm']  # Maturity of the new bond
        initial_spot_rate = bootstrap_forward_rate(forward_curve, T_i)

        # Add this initial spot rate to the list of estimated spot rates
        estimated_spot_rates.append(initial_spot_rate)

        # Create a new combined dataframe of zero-coupons + the new coupon bond
        cf_data = pd.concat([coupon_zeros, coupon_bonds.iloc[[i]]], ignore_index=True)
        cf_data['Spot'] = estimated_spot_rates  # Update with current spot rates

        # Re-run the optimization, using the previous forward curve coefficients as the initial guess
        previous_coefficients = np.concatenate([forward_curve, np.zeros(5)])  # Add a new segment
        result = run_optimization(cf_data, initial_guess=previous_coefficients)
        forward_curve = result.x  # Store the new forward curve coefficients

        # Iteratively adjust the spot rate until bond is priced correctly
        tolerance = 1e-9
        while True:
            new_bond_price = calculate_bond_price(forward_curve, new_bond)

            if abs(new_bond_price - new_bond['Price']) < tolerance:
                break  # Exit the loop if the bond is priced correctly

            dP_dy = compute_price_sensitivity(forward_curve, new_bond)
            initial_spot_rate += (new_bond['Price'] - new_bond_price) / dP_dy

            # Update spot rates and re-run optimization
            estimated_spot_rates[-1] = initial_spot_rate
            cf_data['Spot'] = estimated_spot_rates  # Update spot rates
            result = run_optimization(cf_data, initial_guess=forward_curve)  # Re-run with updated guess
            forward_curve = result.x  # Update forward curve coefficients

    print("Iterative optimization complete for all bonds.")
    return result