import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
today = pd.to_datetime(date.today())
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None


def clean_data(bondData, sheet, today):
    today = pd.Timestamp.today() if today is None else today
    bondData = bondData[sheet]
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
    bondData = bondData.drop(columns=['Ticker', 'Coupon Frequency', 'Price Accrued Interest Flag'])
    bondData['Ttm'] = ((pd.to_datetime(bondData['Maturity Date'], dayfirst=True, errors='coerce') - today).dt.days / 365).round(3)
    bondData['Coupon'] = bondData['Coupon'] / 2
    bondData['Coupon'] = pd.to_numeric(bondData['Coupon'], errors='coerce')
    bondData[~bondData['Yield to Maturity'].isna()]
    bondData['Yield to Maturity'] = bondData['Yield to Maturity'] / 100
    # Remove rows where column 'Dirty Price' is not a number
    bondData = bondData[~bondData['Dirty Price'].astype(str).str.contains('[a-zA-Z]', na=False)]
    # Drop first bond as it's usually volatile
    # bondData = bondData.iloc[1:].reset_index(drop=True)

    # Remove '/d' from description end
    bondData['Description'] = bondData['Description'].str.rstrip('/d')
    bondData['Description'] = bondData['Description'].str.replace(r'(\d{2})(\d{2})$', r'\1/20\2', regex=True)

    return bondData
    breakpoint()


def bond_cleaner(bondData):
    bondData['Maturity Date'] = pd.to_datetime(bondData["Maturity Date"], format='%d-%m-%Y', dayfirst=True)
    bondData['Issue Date'] = pd.to_datetime(bondData["Issue Date"], format='%d-%m-%Y',dayfirst=True)
    # Convert the 'Next Coupon Date' column to ensure consistent handling of NaT values
    bondData['Next Coupon Date'] = pd.to_datetime(bondData['Next Coupon Date'], format='%d-%m-%Y', errors='coerce')
    bondData['Next Coupon Date'] = bondData['Next Coupon Date'].replace(['None', 'NA', '', ' '], np.nan)
    today = pd.Timestamp.today()

    bondData['ISIN'] = bondData['ISIN'].apply(lambda x: 'IT' + ''.join([str(np.random.randint(0, 10)) for _ in range(10)]) if pd.isna(x) else x)

    return bondData


def find_on_the_run(bondData):
    on_run_list = [3,5,7,10,15,20,50]
    bondData_c = bondData.copy()


    bondData_c['Issue Date'] = pd.to_datetime(bondData['Issue Date'], errors='coerce')
    bondData_c['Ttm'] = pd.to_numeric(bondData['Ttm'], errors='coerce')
    #We remove the zeros and re-add them later
    pure_zeros = bondData_c[(bondData_c['Coupon'] == 0) & (bondData_c['Next Coupon Date'].isna())]
    coupon_zeros = bondData_c[(pd.isna(bondData_c['Next Coupon Date'])) & (bondData_c['Coupon'] != 0)]
    zeros = pd.concat([pure_zeros, coupon_zeros], axis=0)

    # We arbitrarily select the second and the last. The second, because the first is too close to maturity.
    # The last, because it carries some value as the last zcb available
    # Select first, second, and last zero-coupon bond
    if len(zeros) > 2:
        zeros = pd.concat([zeros.iloc[[1]], zeros.iloc[[-1]]])
    elif len(zeros) == 2:
        zeros = zeros  # Keep both if only two exist
    elif len(zeros) == 1:
        zeros = zeros  # Keep the only available one
    else:
        zeros = pd.DataFrame()

    selected_bonds = []

    for target_ttm in on_run_list:
        lower_bound = target_ttm * 0.8
        upper_bound = target_ttm * 1.2

        # Filter bonds within the range
        valid_bonds = bondData_c[
            (bondData_c['Ttm'] >= lower_bound) &
            (bondData_c['Ttm'] <= upper_bound)
            ].dropna(subset=['Issue Date'])

        if not valid_bonds.empty:
            # Select the most recently issued bond in this range
            most_recent_bond = valid_bonds.sort_values(
                by='Issue Date',
                ascending=False
            ).head(1)  # Keep it as a DataFrame, not a Series

            selected_bonds.append(most_recent_bond)
        else:
            print(f"No valid bond found for target Ttm = {target_ttm} (Range: {lower_bound:.2f} to {upper_bound:.2f})")

    # Combine selected bonds into a DataFrame
    if selected_bonds:
        bondData_c = pd.concat(selected_bonds).sort_values(by=['Ttm']).reset_index(drop=True)
        bondData_c.columns.name=None
    else:
        bondData_c = pd.DataFrame()
    logger.info(f"Bond dataframe reduced from length  {len(bondData)} to length {len(bondData_c)}.")
    bondData = bondData_c.copy()
    logger.info(f"{bondData}")
    return bondData, zeros

