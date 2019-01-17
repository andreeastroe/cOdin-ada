import utilities.preprocessor as pp
import utilities.visual as vis
import analysisMethods.PCA as pca
import analysisMethods.EFA as efa
import analysisMethods.CCA as cca
import analysisMethods.HCA as hca
import analysisMethods.LDA as lda
import pandas as pd

class ada:

    def __init__(self, filePath, filePath2 = None, method = ""):
        self.data = pp.readDataframeFromCSV(filePath, replaceNA = True)
        if filePath2 is not None:
            self.data2 = pp.readDataframeFromCSV(filePath2, replaceNA=True)
        self.method = method

        # instantiate a data analysis method class according to the method parameter
        if self.method == "PCA":
            self.PCA = pca.PCA(self.data)
            R = self.PCA.getCorrelation()
            alpha = self.PCA.getEigenValues()
            a = self.PCA.getEigenVectors()
            Rxc = self.PCA.getCorrelationFactors()
            C = self.PCA.getPrincipalComponents()

            obs_name = self.data.index
            var_name = self.data.columns[1:]

            # def writeDataFrameToCSV(data, filePath="data.csv", cols=None, index=None):

            # Save the processed data
            dataTab = pd.DataFrame(data=self.data, columns=var_name, index=obs_name)
            pp.writeDataFrameToCSV(dataTab, filePath = "data.csv")

            # Save the correlation matrix
            correlMatrix = pd.DataFrame(data=R, columns=var_name, index=var_name)
            pp.writeDataFrameToCSV(correlMatrix, filePath="CorrelationMatrix.csv")

            # Show the correlogram
            vis.correlogram(correlMatrix)

            # Save the correlation factors
            m = self.data.shape[1]  # number of variables
            correlFactors = pd.DataFrame(Rxc, index=var_name, columns=
            ["C" + str(k + 1) for k in range(m)])
            pp.writeDataFrameToCSV(correlFactors, filePath="CorrelationFactors.csv")

            # Show factors correlogram
            vis.correlogram(correlFactors, "Factors Correlogram")
            vis.corrCircles(correlFactors, 0, 1)

            # Show the eigenvalues graphic
            vis.eighenValues(alpha)
        elif self.method == "EFA":
            self.EFA = efa.EFA(self.data, input("Enter significance threshold for scores: "))
            pp.writeDataFrameToCSV(self.EFA.getLoadings(), "EFALoadings.csv")
            pp.writeDataFrameToCSV(self.EFA.getRotatedLoadings(), "EFARotatedLoadings.csv")
            pp.writeDataFrameToCSV(self.EFA.getEigenValues(), "EFAEigenvalues.csv")
            vis.correlogram(self.EFA.KMO_test[0], " KMO Indices, KMO Total = "+ self.EFA.KMO_test[1])
            vis.correlogram(self.EFA.getLoadings(), "Factorial Coefficients")
            vis.correlogram(self.EFA.getRotatedLoadings(), "Quartimax Rotated Factorial Coefficients")
        elif self.method == "CCA":
            self.CCA = cca.CCA(self.data)
            pp.writeDataFrameToCSV(self.cca.set1, "Set1.csv")
            pp.writeDataFrameToCSV(self.cca.set2, "Set2.csv")
            pp.writeDataFrameToCSV(self.cca.x_loadings_, "Set1FactorLoadings.csv")
            pp.writeDataFrameToCSV(self.cca.y_loadings_, "Set2FactorLoadings.csv")
            print("Chi square computed table: ", self.cca.chi2_computed_table)
            vis.correlogram(self.cca.hi2_computed_table, title = "Bartlett-Wilks significance test")
            print("Chi square estimated table: ", self.cca.chi2_estimated_table)
            vis.correlogram(self.cca.chi2_estimated_table, title = "Bartlett-Wilks significance test")
        elif self.method == "LDA":
            self.LDA = lda.LDA(self.data, self.data2)
            pp.writeDataFrameToCSV(self.LDA.getTableClassificationBErr(), "BClassificationError.csv")
            pp.writeDataFrameToCSV(self.LDA.getTableClassificationB(), "BClassification.csv")
            pp.writeDataFrameToCSV(self.LDA.getTableClassification(), "Classification.csv")
            if self.LDA.getNumberOfDiscriminantAxes() > 1:
                vis.scatter_discriminant(self.LDA.getX()[:, 0], self.LDA.getX()[:, 1], self.LDA.getSet2(), self.LDA.getSetX.index, self.LDA.getXc[:, 0], self.LDA.getXc[:, 1], self.LDA.getLdaModel().classes_)
            for num in range(self.LDA.getNumberOfDiscriminantAxes()):
                vis.distribution(self.LDA.getX()[:, num], self.LDA.getSet2(), self.LDA.getLdaModel().classes_, axa=num)
        else: # self.method == "HCA"
            self.HCA = hca.HCA(self.data)
            # do shit
