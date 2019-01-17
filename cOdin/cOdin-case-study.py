# create a demo application containing the necessary data sets in order to demonstrate the library functionality (0.5 points)
import ada

filePath = 'Teritorial.csv'
filePath2 = 'student-por.csv'
filePath3 = "MortalityEU.csv"
filePath3 = "Energy.csv"
filePath4 = "ProiectB.csv"
filePath5 = "ProiectBEstimare.csv"
filePath6 = "datasetIndicatori.csv"

# Proof of Principal Component Analysis
print("*******************Principal Component Analysis************************")
ada.ADA(filePath, method = 'PCA')

# Proof of Explanatory Factor Analysis
print("*******************Explanatory Factor Analysis************************")
ada.ADA(filePath2, method = 'EFA')

# Proof of Canonical Correlation Analysis
print("*******************Canonical Correlation Analysis************************")
ada.ADA(filePath3, method = 'CCA')

# Proof of Linear Discriminant Analysis
print("*******************Linear Discriminant Analysis************************")
# ada.ADA(filePath = filePath4, filePath2= filePath5, method = 'LDA')

# Proof of Hierarchical Cluster Analysis
print("*******************Hierarchical Cluster Analysis************************")
# ada.ADA(filePath6, method = 'HCA')


