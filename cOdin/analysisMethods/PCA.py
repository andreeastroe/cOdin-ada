# class for holding a Principal Component(PCA) problem
import pandas as pd
import numpy as np
import utilities.preprocessor as pp


class PCA:

    # Initializer / Instance Attributes
    def __init__(self, X):
        if type(X) is pd.DataFrame:  # if it is a DataFrame, we convert it to ndarray
            values = X.iloc[:, 1:].values
            X = values
        self.X = X

        print("Reached PCA")
        self.R = np.corrcoef(self.X, rowvar=False)  # Compute the correlation matrix
        self.eigenVal, self.eigenVect = np.linalg.eigh(self.R)

        # Sort the eigenvalues and the corresponding eigenvectors in descending order
        ev_list = zip(self.eigenVal, self.eigenVect)
        ev_list.sort(key=lambda tup: tup[0], reverse=True)
        self.alpha, self.a = zip(*ev_list)
        pp.inverse(self.a)

        # Compute the correlation factors
        self.Rxc = self.a * np.sqrt(self.alpha)

        # Compute the principal components for standardized X
        self.Xstd = pp.standardize2(X)
        self.C = self.Xstd @ self.a

    # Return the correlation matrix of the initial (causal) variables
    def getCorrelation(self):
        return self.R

    # Return the eigenvalues of the correlation matrix
    def getEigenValues(self):
        return self.alpha

    # Return the eigenvectors of the correlation matrix
    def getEigenVectors(self):
        return self.a

    def getCorrelationFactors(self):
        return self.Rxc

    def getPrincipalComponents(self):
        return self.C
