# class for holding a Linear Discriminant (LDA) problem
import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as disc
import utilities.preprocessor as pp


class LDA:

    # Initializer / Instance Attributes
    def __init__(self, X, Y): # X, Y are DataFrames
        var = np.array(X.columns)

        x = [int(i) for i in
             input("Enter indices for columns used for categorization: ").split()]  # enter numbers with spaces
        if not all(isinstance(el, int) for el in x):
            raise ValueError('Please input only integer values.')

        var_categorical = var[np.asarray(x)]
        pp.coding(X, var_categorical)
        pp.coding(Y, var_categorical)
        self.X = X
        self.Y = Y

        # Select the predictor variables and the discriminant variable
        x = [int(i) for i in input("Enter indices for discriminant variable columns : ").split()] # enter numbers with spaces
        if not all(isinstance(el, int) for el in x):
            raise ValueError('Please input only integer values.')
        self.var_p = var[list(set(range(X.shape[1])) - set(x))] # columns for predictor variables
        self.var_c = var[np.asarray(x)] # columns for discriminant variables
        print(self.var_p)
        print(self.var_c)

        self.set1 = X[self.var_p].values
        self.set2 = X[self.var_c].values

        # we start using sklearn.discriminant_analysis
        self.lda_model = disc.LinearDiscriminantAnalysis()
        self.lda_model.fit(self.set1, self.set2)

        setBaseClass = self.lda_model.predict(self.set1)
        self.table_classificationB = pd.DataFrame(
            data={str(self.var_c[0]): self.set2, 'prediction': setBaseClass},
            index=self.set1.index)
        self.table_classificationB_err = self.table_classificationB[self.set2 != setBaseClass]
        len = len(self.set2)
        len_err = len(self.table_classificationB_err)
        self.degreeOfCredence = (len - len_err) * 100 / len

        setTestClass = self.lda_model.predict(Y[self.var_p].values)
        self.table_of_classification = pd.DataFrame( data={'prediction': setTestClass}, index=Y.index)

        g = self.lda_model.classes_
        q = len(g)
        Cmat = pd.DataFrame(data=np.zeros((q, q)), index=g, columns=g)
        for i in range(len):
            Cmat.loc[Y[i], setBaseClass[i]] += 1
        accuracy_groups = np.diag(Cmat) * 100 / np.sum(Cmat, axis=1)

        scalings = self.lda_model.scalings_ # Instances on the first 2 axes of discrimination
        self.x = self.set1 @ scalings
        means = self.lda_model.means_
        self.xc = means @ scalings

        self.noDiscriminantAxes = np.shape(scalings) # number of discriminant axes equal to the number of columns in len

    # Return the columns  of the predictor variables
    def getPredictorColumns(self):
        return self.var_p

    # Return the columns  of the predictor variables
    def getDiscriminantColumns(self):
        return self.var_c

    # Return the umber of discriminant axex
    def getEigenVectors(self):
        return self.noDiscriminantAxes

    def getLdaModel(self):
        return self.lda_model

    def getDegreesOfCredence(self):
        return self.degreeOfCredence

    def getTableClassificationBErr(self):
        return self.table_classificationB_err

    def getTableClassificationB(self):
        return self.table_classificationB

    def getTableClassification(self):
        return self.table_of_classification

    def getNumberOfDiscriminantAxes(self):
        return self.noDiscriminantAxes

    def getX(self):
        return self.x

    def getXc(self):
        return self.xc

    def getSet1(self):
        return self.set1

    def getSet2(self):
        return self.set2

    def getX(self):
        return self.X

    def getY(self):
        return self.Y
