#  implement the input and output data files processing (0.5 point)
#  provide data transformation functionality, such as matrix centering, standardization, filling-in the NA, NaN cells etc. (0.5 points)
import pandas as pd
import numpy as np

def readDataframeFromCSV(filePath, replaceNA = False):
    try:
        data = pd.read_csv(filePath, index_col=1);
        obs_name = data.index   # stores the observations' names from the data
        var_name = data.columns[1:]     # stores the variables' names from the data
        values = data[var_name].values  # get the actual values from the data for the variable names
        if replaceNA == True:   # if the replaceNA parameter is specified with the value True, it makes a call to the replaceNAValues function
            replaceNAValues(values)
            data[var_name].values = values  #sets back the cleaned values in the data variable

    except pd.errors.EmptyDataError: #Exception that is thrown in pd.read_csv (by both the C and Python engines) when empty data or header is encountered.
        print("File not found or path is incorrect. Make sure it is a .csv file and exists at the specified destination. ")
    finally:
        print("exit")
    return data

# Replaces NA values in a numpy.ndarray with the average value on the column they belong to
def replaceNAValues(values):
    averages = np.nanmean(values, axis=0)   # Computes the arithmetic mean along the specified axis, ignoring NaNs
    positions = np.where(np.isnan(values)) # array of positions with NA values
    values[positions] = averages[positions[1]]
    return values

# Standardize the column (variable) values for a pandas.DataFrame - luat de la Vinte
def standize(values):
    averages = np.mean(values, axis = 0 )   # computes average values for the columns
    stdandardDevs = np.std(values, axis = 0)    # computes standard deviations for the values in the columns
    Xstd = (values - averages) / stdandardDevs  # standardizes each value in the initial data input
    return Xstd