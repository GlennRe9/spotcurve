# Bond Pricing and Forward Curve Construction

This project implements a pricing framework for bonds and a forward curve construction methodology using piecewise quartic polynomial interpolation (PQPI). It provides functionality to handle zero-coupon and coupon bonds, compute spot and forward rates, and apply perturbation-based optimisation techniques to ensure accurate pricing.

---

## Inspiration

The methods implemented in this project are inspired by the paper:
**"Computing maximally smooth forward rate curves for coupon bonds: An iterative piecewise quartic polynomial interpolation method"** by *Paul M. Beaumont* and *Yaniv Jerassy-Etzion*.

---

## Features

- **Data Cleaning**: Cleans and preprocesses bond data to ensure consistency.
- **Bootstrap Spot Rates**: Implements both discrete and continuous compounding for zero-coupon bonds.
- **Forward Curve Construction**:
  - Piecewise quartic polynomial interpolation (PQPI).
  - Linear regression for global smoothing.
- **Perturbation Algorithm**: Iteratively adjusts the spot rate to achieve accurate bond pricing.
- **Custom Constraints**: Builds constraints for pricing, continuity, and differentiability based on bond characteristics.
- **Flexible Compounding**: Supports discrete and continuous compounding methods.
- **Visualisation**: Plots the spot and forward curves for analysis.

---

## Prerequisites

- Python 3.8 or later
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `scipy`

Install the dependencies using pip:
bash
pip install numpy pandas matplotlib scikit-learn scipy



Files and Structure

	•	main.py: The main entry point for running the program.
	•	cleaning.py: Functions for data cleaning and preprocessing.
	•	constraints.py: Builds the pricing and continuity constraints for the optimisation problem.
	•	optimisation.py: Implements the PQPI and perturbation-based optimisation algorithms.
	•	bond_calculator.py: Contains utility functions for bond calculations like cash flow generation and pricing.
	•	README.md: This file.


Usage

	1.	Prepare the Input Data:
	•	Ensure your bond data is in a structured format (e.g., .xlsx or .csv).
	•	The data should include fields like Maturity Date, Issue Date, Coupon, Yield to Maturity, Dirty Price, and Time to Maturity (Ttm).
	2.	Run the Program:
Execute the main.py script:

```python main.py```


	3.	Configuration:
Modify the compounding flag in the script to switch between discrete and continuous compounding:
compounding = "Cont"  # Options: "Discrete" or "Cont"

	4.	Output:
	•	The program generates forward and spot curves.
	•	Logs key steps, including the perturbation process, pricing adjustments, and optimisation results.

Example Workflow

	1.	Load bond data into a DataFrame using pandas.
	2.	Clean and preprocess the data using cleaning.py.
	3.	Bootstrap initial spot rates for zero-coupon bonds.
	4.	Iteratively:
	•	Add coupon bonds to the calculation.
	•	Adjust spot rates using the perturbation method.
	•	Optimise the forward curve using PQPI.
	5.	Visualise the results using matplotlib.

Key Functions

Data Cleaning

	•	cleaning.clean_data(bondData): Prepares the bond data for analysis.

Spot and Forward Rate Calculation

	•	bond_calculator.bootstrap_spot(...): Bootstraps spot rates for bonds.
	•	bond_calculator.calculate_price_from_forward_curve(...): Calculates bond prices from the forward curve.

Optimisation

	•	optimisation.run_optimisation(...): Runs the PQPI-based optimisation to construct the forward curve.
	•	optimisation.build_pricing_constraints(...): Builds the constraint matrices for the optimisation problem.

Perturbation

	•	perturbation_function(...): Iteratively adjusts spot rates to achieve accurate pricing.

Visualisation

The program includes plotting functions to visualise the spot and forward curves. Adjustments for plotting can be made in the main.py script.

Future Enhancements

	•	Add support for additional bond types (e.g., floating-rate bonds).
	•	Enhance visualisation for better interpretability.
	•	Implement advanced interpolation methods beyond PQPI.
	•	Add unit tests for core functionalities.


Contact

For any queries or contributions, please reach out to glennregis@yahoo.com

