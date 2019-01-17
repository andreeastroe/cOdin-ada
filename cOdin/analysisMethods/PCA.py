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

        self.R = np.corrcoef(self.X, rowvar=False)  # Compute the correlation matrix
        self.eigenVal, self.eigenVect = np.linalg.eigh(self.R)

        # Sort the eigenvalues and the corresponding eigenvectors in descending order
        k_reverse = [k for k in reversed(np.argsort(self.eigenVal))]
        self.alpha = self.eigenVal[k_reverse]
        self.a = self.eigenVect[:, k_reverse]
        for i in range(len(self.alpha)):
            minim = np.min(self.a[:, i])
            maxim = np.max(self.a[:, i])
            if np.abs(minim) > np.abs(maxim):
                self.a[:, i] = -self.a[:, i]

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
