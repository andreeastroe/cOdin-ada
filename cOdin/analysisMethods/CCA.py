# class for holding a Canonical Correlation(CCA) problem
import pandas as pd
import numpy as np
import utilities.visual as vis
import sklearn.cross_decomposition as skl
import utilities.preprocessor as pp
import utilities.statistic as st


class CCA:

    # Initializer / Instance Attributes
    def __init__(self, X):
        self.X = X

        aux = input("Enter delimiting index between datasets: ")
        if not isinstance(int(aux), int):
            raise ValueError('Please input only integer values.')
        x=int(aux)

        columns = X.columns[1:]
        set1_columns = columns[:x]
        set2_columns = columns[x:]
        self.set1 = X[set1_columns].values
        self.set2 = X[set2_columns].values
        # write them to files in ada

        n = self.set1.shape[0]
        m = self.set1.shape[1]
        p = np.shape(self.set2)[1]
        self.noComponents = min (m, p)

        self.cca = skl.CCA(n_components = self.noComponents)
        self.cca.fit(self.set1, self.set2)

        # Compute factor loadings for both sets
        self.x_loadings_ = self.cca.x_loadings_
        self.y_loadings_ = self.cca.y_loadings_

        # Compute canonical scores for both sets
        self.x_scores = self.cca.x_scores_ # when written to file, separate values with "," to obtain a csv format
        self.y_scores = self.cca.y_scores_ # when written to file, separate values with "," to obtain a csv format

        # Compute the canonical correlation coefficients
        self.correlCoeffs = np.array([np.corrcoef(self.x_scores[:, i], self.y_scores[:, i], rowvar=False)[0, 1] for i in range(self.noComponents)])

        chi2_computed, chi2_estimated = st.bartlett_wilks(self.correlCoeffs, n, m, p, self.noComponents)

        self.chi2_computed_table = pd.DataFrame(chi2_computed, index=['r' + str(i) for i in range(1, self.noComponents + 1)], columns=['chi2_computed'])
        vis.correlogram(self.chi2_computed_table, "Bartlett-Wilks significance test", 0) # get in ada

        self.chi2_estimated_table = pd.DataFrame(chi2_estimated, index=['r' + str(i) for i in range(1, m + 1)], columns=['chi2_estimated'])
        vis.correlogram(self.chi2_estimated_table, "Bartlett-Wilks significance test", 0) # get in ada

    # Return the correlation coefficients matrix
    def getCorrelationCoefficients(self):
        return self.correlCoeffs

    # Return the canonical scores for the first dataset
    def getXScores(self):
        return self.x_scores

    # Return the canonical scores for the second dataset
    def getYScores(self):
        return self.y_scores

    # Return the factor loadings for the first dataset
    def getXLoadings(self):
        return self.x_loadings_

    # Return the factor loadings for the second dataset
    def getYLoadings(self):
        return self.y_loadings_

    # Return the first dataset
    def getSet1(self):
        return self.set1

    # Return the second dataset
    def getSet2(self):
        return self.set2

    # Return the cca component
    def getCca(self):
        return self.cca

    # Return the Chi Square Computed Table
    def getChiSquareComputedTable(self):
        return self.chi2_computed_table

    # Return the Chi Square Estimated Table
    def getChiSquareEstimatedTable(self):
        return self.chi2_estimated_table