#  implement the input and output data files processing (0.5 point)
#  provide data transformation functionality, such as matrix centering, standardization, filling-in the NA, NaN cells etc. (0.5 points)
import pandas as pd
import numpy as np
import scipy as sp

# Replaces NA values in a numpy.ndarray with the average value on the column they belong to
def replaceNAValues(values):
    averages = np.nanmean(values, axis=0)   # Computes the arithmetic mean along the specified axis, ignoring NaNs
    positions = np.where(np.isnan(values)) # array of positions with NA values
    values[positions] = averages[positions[1]]
    return values

# Replaces NA values in a numpy.ndarray with the mean/mode value on the column they belong to
def replaceNANValuesDataFrame(values):
    for c in values.columns:
        if pd.api.types.is_numeric_dtype(values[c]):
            averages = np.nanmean(values, axis=0)  # Computes the arithmetic mean along the specified axis, ignoring NaNs
            positions = np.where(np.isnan(values))  # array of positions with NA values
            values[positions] = averages[positions[1]]
            return values
        else:
            #nu garantez ca merge
            if values[c].isna().any():
                values = values[~np.isnan(values)]
                mode = sp.stats.mode(values[c])
                # mode = values[c].mode()
                values[c] = values[c].fillna(mode[0])

# Reads data from a .csv file to a pandas.DataFrame
def readDataframeFromCSV(filePath, replaceNA = False, index_col = 1):
    try:
        data = pd.read_csv(filePath, index_col)
        print(data)
        print(data[data.columns[1:]])
        obs_name = data.index   # stores the observations' names from the data
        var_name = data.columns[1:]     # stores the variables' names from the data
        values = data[var_name].values  # get the actual values from the data for the variable names
        if replaceNA == True:   # if the replaceNA parameter is specified with the value True, it makes a call to the replaceNAValues function
            replaceNAValues(values)
            data[var_name].values = values  #sets back the cleaned values in the data variable

        print("Initial dataset: \n" + data)

        n = data.shape[0]  # number of observation
        m = data.shape[1]  # number of variables
        print("Contains " + n + " observations and " + m + " variables.")

    except pd.errors.EmptyDataError: #Exception that is thrown in pd.read_csv (by both the C and Python engines) when empty data or header is encountered.
        print("File not found or path is incorrect. Make sure it is a .csv file and exists at the specified destination. ")
    finally:
        print("exit")
    return data

# Reads data from a .txt file to a pandas.DataFrame
def readDataframeFromTXT(filePath, replaceNA = False):
    try:
        data = pd.read_fwf(filePath, index_col=1);
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

# Writes a pandas.DataFrame or numpy.ndarray to a .csv file
def writeDataFrameToCSV(data, filePath = "data.csv", cols = None, index = None):
    if type(data) is pd.DataFrame:
        data.to_csv(path_or_buf = filePath)
        print("Data successfully saved to " + filePath)
    else:
        datadf = pd.DataFrame(data)
        if cols is not None:    # if there is a specified ndarray with the columns, use it
            datadf.columns = cols
        if index is not None:   # if there is a specified ndarray with the index, use it
            datadf.index = index
        datadf.to_csv(path_or_buf=filePath)
        print("Data successfully saved to " + filePath)

# Used in EFA
def evaluate(mat, correl, alpha):
    n = np.shape(mat)[0] # number of rows
    scores = mat / np.sqrt(alpha)  # compute scores

    matsq = mat * mat # compute cosines
    add = np.sum(matsq, axis=1)
    q = transpose(transpose(matsq) / add)

    beta = matsq / (alpha * n)  # compute distributions

    correlsq = correl * correl # compute commonalities
    communalities = np.cumsum(correlsq, axis=1)
    return scores, q, beta, communalities

# Standardize the column (variable) values for a pandas.DataFrame - luat de la Vinte
def standardize2(values):
    averages = np.mean(values, axis = 0 )   # computes average values for the columns
    stdandardDevs = np.std(values, axis = 0)    # computes standard deviations for the values in the columns
    Xstd = (values - averages) / stdandardDevs  # standardizes each value in the initial data input
    return Xstd

# http://rasbt.github.io/mlxtend/user_guide/preprocessing/standardize/
def standardize(array):
    ary_new = array.astype(float)
    dim = ary_new.shape
    if len(dim) == 1:
        ary_new = ary_new[:, np.newaxis]

    if isinstance(ary_new, pd.DataFrame):
        ary_newt = ary_new.loc
        columns = ary_new.columns
    elif isinstance(ary_new, np.ndarray):
        ary_newt = ary_new
        columns = list(range(ary_new.shape[1]))
    else:
        raise AttributeError('Input array must be a pandas.DataFrame or numpy.ndarray')

    parameters = {'avgs': ary_newt[:, columns].mean(axis=0),
                  'stds': ary_newt[:, columns].std(axis=0, ddof=0)}
    are_constant = np.all(ary_newt[:, columns] == ary_newt[0, columns], axis=0)
    for c, b in zip(columns, are_constant):
        if b:
            ary_newt[:, c] = np.zeros(dim[0])
            parameters['stds'][c] = 1.0

    ary_newt[:, columns] = ((ary_newt[:, columns] - parameters['avgs']) /
                            parameters['stds'])
    return ary_newt[:, columns] # returns pandas.DataFrame, copy of the array or DataFrame with standardized columns

# Used in pca
def inverse(data):
    if type(data) is pd.DataFrame:
        for col in data.columns:
            minim = data[col].min()
            maxim = data[col].max()
            if abs(minim) > abs(maxim):
                data[col] = -data[col]
    else:
        for i in range(np.shape(data)[1]):
            minim = np.min(data[:, i])
            maxim = np.max(data[:, i])
            if np.abs(minim) > np.abs(maxim):
                data[:, i] = -data[:, i]

# Centers a given matrix using the mean. Receives Dataframe or ndarray
def meanCenter(data):
    if type(data) is pd.DataFrame: # if it is a DataFrame, we convert it to ndarray
        values = data.values
    else:
        values = data
    meanValues = values.mean(axis=0)
    values -= meanValues
    return values

# Calculates the inverse of a given matrix. Receives Dataframe or ndarray
def inverse(data):
    if type(data) is pd.DataFrame:  # if it is a DataFrame, we convert it to ndarray
        values = data.values
    else:
        values = data
    inverseValues = np.linalg.inv(values)
    return inverseValues

# Calculates the inverse of a given matrix. Receives Dataframe or ndarray
def transpose(data):
    if type(data) is pd.DataFrame:  # if it is a DataFrame, we convert it to ndarray
        values = data.values
    else:
        values = data
    transposeValues = np.transpose(values)
    return transposeValues

# Used in HCA for building partition tables
def partition(h, k):
    n = np.shape(h)[0] + 1
    g = np.arange(0, n)
    for i in range(n - k):
        k1 = h[i, 0]
        k2 = h[i, 1]
        g[g == k1] = n + i
        g[g == k2] = n + i
    clusters = ['c'+str(i) for i in pd.Categorical(g).codes]
    return clusters