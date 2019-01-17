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

        # si aici apelam mai departe clasa din alea 5 care corespunde cu method
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
            self.EFA = efa.EFA(self.data)
            # do shit
        elif self.method == "CCA":
            self.CCA = cca.CCA(self.data)
            # do shit
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
