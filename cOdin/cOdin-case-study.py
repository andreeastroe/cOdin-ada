# create a demo application containing the necessary data sets in order to demonstrate the library functionality (0.5 points)
import ada

filePath = 'datasets//Teritorial.csv'
filePath2 = 'datasets//student-por.csv'
filePath3 = "datasets//MortalityEU.csv"
filePath3 = "datasets//Energy.csv"
filePath4 = "datasets//ProiectB.csv"
filePath5 = "datasets//ProiectBEstimare.csv"

# Proof of Principal Component Analysis
# ada.ADA(filePath2, method = 'PCA')

# Proof of Explanatory Factor Analysis
# ada.ADA(filePath2, method = 'EFA')

# Proof of Canonical Correlation Analysis
# ada.ADA(filePath3, method = 'CCA')

# Proof of Linear Discriminant Analysis
ada.ADA(filePath3, method = 'LDA')


