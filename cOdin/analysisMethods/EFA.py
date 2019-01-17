# class for holding a Explanatory Factor Analysis (EFA) problem
import pandas as pd
import factor_analyzer as fa
import analysisMethods.PCA as pca
import utilities.preprocessor as pp

class EFA:

    # Initializer / Instance Attributes
    def __init__(self, X, threshold): # assume receiving a DataFrame
        columns = X.columns[1:] # exclude the first column as it contains the observations' names
        index = X.index
        self.X = X[columns].values
        self.pca = pca.PCA(self.X)
        PC = self.pca.getPrincipalComponents()
        a = self.pca.getEigenVectors()
        alpha = self.pca.getEigenValues()
        correl = self.pca.getCorrelation()

        self.scores, q, beta, communalities = pp.evaluate(PC, correl, alpha)
        self.Bartlett_test = fa.calculate_bartlett_sphericity(pd.DataFrame(self.X, index=index, columns=columns))    # compute Bartlett test
        self.KMO_test = fa.calculate_kmo(pd.DataFrame(self.X, index=index, columns=columns))  # compute Kaiser, Meyer, Olkin test as a measure of sampling adequacy
        if self.KMO_test[1] < float(threshold):
            print("No significant factor found!")
            exit(1)

        self.fa = fa.FactorAnalyzer()
        self.fa.analyze(pd.DataFrame(self.X, index=index, columns=columns), rotation=None)
        self.loadings = self.fa.loadings

        self.fa.analyze(pd.DataFrame(self.X, index=index, columns=columns), rotation='quartimax')
        self.rotatedLoadings = self.fa.loadings

        self.eigenValues = self.fa.get_eigenvalues()


    # Return the computed scores
    def getScores(self):
        return self.scores

    # Return the computed scores
    def getDatasetValues(self):
        return self.X

    # Return the computed scores
    def getBartlettTestResults(self):
        return self.Bartlett_test

    # Return the computed scores
    def getKMOTestResults(self):
        return self.KMO_test

    # Return the computed scores
    def getLoadings(self):
        return self.loadings

    # Return the computed scores
    def getRotatedLoadings(self):
        return self.rotatedLoadings

    # Return the computed scores
    def getEigenValues(self):
        return self.eigenValues
