# class for holding a Hierarchical Cluster(HCA) problem
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as ch
import scipy.spatial.distance as spad
import utilities.preprocessor as pp


class HCA:

    # Initializer / Instance Attributes
    def __init__(self, X):
        self.columns = X.columns[1:] # assume first column is for observations names
        index = X.index
        self.values = X[self.columns].values
        self.obsMethod = input("Specify linkage method for observations (single, complete, average, centroid, median, ward, weighted) :")
        if self.obsMethod not in list(ch._LINKAGE_METHODS):
            raise ValueError('Please input a value in the specified set')

        # Classification of observations
        self.obs = ch.linkage(self.values, method=self.obsMethod, metric='braycurtis')

        self.partitions = pd.DataFrame(index=index) # building the partition table

        noRows = np.shape(self.obs)[0]
        val = np.argmax(self.obs[1:noRows, 2] - self.obs[:(noRows - 1), 2])
        n = noRows - val
        optimalGroups = pp.partition(self.obs, n)
        self.partitions['Optimal_Partition'] = optimalGroups

        self.optimalThreshold = (self.obs[val,2]+self.obs[val+1,2])/2
        # plot dendogram in ada & maybe write data in csv

        self.noGroups = input("Specify desired number of groups :")
        groups = pp.partition(self.obs, self.noGroups)
        self.partitions['Partition_' + str(self.noGroups)] = groups
        self.threshold = (self.obs[noRows-n,2]+ self.obs[noRows-n+1,2])/2
        # plot dendogram in ada & maybe write data in csv

        # Classification of variables
        self.varsMethod = input("Specify linkage method for variables (single, complete, average, centroid, median, ward, weighted) :")
        if self.varsMethod not in list(ch._LINKAGE_METHODS):
            raise ValueError('Please input a value in the specified set')
        self.vars = ch.linkage(pp.transpose(self.values), method=self.varsMethod, metric='braycurtis')
        # plot dendogram in ada


    # Return the dataset values used
    def getDatasetValues(self):
        return self.values

    # Return the computed observations' classification
    def getObservationsClassification(self):
        return self.obs

    # Return the computed variables' classification
    def getVariablesClassification(self):
        return self.vars

    # Return the computed threshold
    def getThreshold(self):
        return self.threshold

    # Return the computed optimal threshold
    def getOptimalThreshold(self):
        return self.optimalThreshold

    # Return the specified number of groups
    def getNumberOfGroups(self):
        return self.noGroups

    # Return the computed partitions
    def getPartitions(self):
        return self.partitions