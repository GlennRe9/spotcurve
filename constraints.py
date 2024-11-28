import numpy as np
import pandas as pd

def build_pricing_constraints(clean_data, compounding):
    """
    Constructs the pricing, continuity, and differentiability constraints for the optimization.

    Parameters:
    clean_data (DataFrame): Dataframe with multiple bonds and their relevant data.
                            Must contain columns: 'Ttm' (Time-to-Maturity) and 'Spot'.

    Returns:
    A (numpy array): Constraint matrix (combines pricing, continuity, and differentiability constraints).
    B (numpy array): Right-hand side values for the constraints.
    """
    # Sort the dataframe by 'Ttm' (Time-to-Maturity)
    clean_data = clean_data.sort_values(by='Ttm')
    Ttm = clean_data['Ttm'].values
    df = clean_data['df'].values

    # Number of bonds (nodes)
    m = len(clean_data)
    n = m - 1
    n_rows = (m * 4) + 5
    n_cols = 5 * (m + 1) # Including additional segment

    # Initialize constraint matrices
    A = np.zeros((n_rows, n_cols))
    B = np.zeros(n_rows)

    discount_factors = np.array(clean_data['df'].values, dtype=np.float64)
    # Assume the discount factor at T0 is 1
    assert(len(discount_factors) == m)
    discount_factors = np.insert(discount_factors, 0, 1.0)  # Insert 1 at the beginning
    # Pricing constraints (ensuring P(T_i) = P_hat(T_i))
    # We equal the polynomial to the forward rate, which is calculated by dividing each discount factor by its previous

    for i in range(m):
        T_i = clean_data.iloc[i]['Ttm']
        d_t = clean_data.iloc[i]['Delta_t']
        # Pricing constraint matrix
        A[i, 5 * i: 5 * (i + 1)] = [T_i ** 4, T_i ** 3, T_i ** 2, T_i, 1]

        if compounding == "Cont":
            # Use log discount factor ratios for continuous compounding
            B[i] = np.log(discount_factors[i] / discount_factors[i + 1]) / d_t
        elif compounding == "Discrete":
            # Use forward rate formula for discrete compounding
            B[i] = (discount_factors[i] / discount_factors[i + 1]) ** (1 / d_t) - 1
        else:
            raise ValueError("Invalid compounding method. Choose 'Discrete' or 'Cont'.")

    # This is the log way which we have now left
    #B[:m] = np.log(discount_factors[1:] / discount_factors[:-1])

    # Continuity constraints at the nodes
    # We make sure that the poly calculated from the left is equal to the poly calculated from the right
    for i in range(0, m):
        T_i = clean_data.iloc[i]['Ttm']
        row = m + i

        # Continuity constraint: f_{i}(T_i) = f_{i+1}(T_i)
        A[row, 5 * i: 5 * (i + 1)] = [T_i ** 4, T_i ** 3, T_i ** 2, T_i, 1]
        A[row, 5 * (i + 1): 5 * (i + 2)] = [-T_i ** 4, -T_i ** 3, -T_i ** 2, -T_i, -1]

    # Differentiability constraints (first derivative continuity at nodes)
    for i in range(0, m):
        T_i = clean_data.iloc[i]['Ttm']
        row = m * 2 + i

        # First derivative continuity: f'_{i}(T_i) = f'_{i+1}(T_i)
        A[row, 5 * (i): 5 * (i+1)] = [4 * T_i ** 3, 3 * T_i ** 2, 2 * T_i, 1, 0]
        A[row, 5 * (i+1): 5 * (i + 2)] = [-4 * T_i ** 3, -3 * T_i ** 2, -2 * T_i, -1, 0]

    # Second derivative continuity at nodes
    for i in range(0, m):
        T_i = clean_data.iloc[i]['Ttm']
        row = m * 3 + i

        # Second derivative continuity: f''_{i}(T_i) = f''_{i+1}(T_i)
        A[row, 5 * (i): 5 * (i+1)] = [12 * T_i ** 2, 6 * T_i, 2, 0, 0]
        A[row, 5 * (i+1): 5 * (i + 2)] = [-12 * T_i ** 2, -6 * T_i, -2, 0, 0]

    # Boundary conditions for the forward curve
    A[-5, 4] = 1   # f(0) = y(0)
    B[-5] = clean_data.iloc[0]['Spot']  # Initial spot rate
    # Terminal conditions for the derivatives at T_m
    T_m = clean_data.iloc[-1]['Ttm']
    # Boundary conditions to enforce a flat curve beyond T_m
    A[-4, -5] = 1  # a_{m+1} = 0
    A[-3, -4] = 1  # b_{m+1} = 0
    A[-2, -3] = 1  # c_{m+1} = 0
    A[-1, -2] = 1  # d_{m+1} = 0

    A_df = pd.DataFrame(A)
    B_df = pd.DataFrame(B)
    return A, B

