import utilities.preprocessor as pp
import analysisMethods.PCA

class ada:

    def __init__(self, filePath, method = ""):
        self.data = pp.readDataframeFromCSV(filePath, replaceNA = True)
        self.method = method

        # si aici apelam mai departe clasa din alea 5 care corespunde cu method
