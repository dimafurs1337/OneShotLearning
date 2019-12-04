import numpy as np
from sklearn.metrics import accuracy_score

class Accuracy:
    @staticmethod
    def compute_accuracy(y_test, prediction):
        def filter_values(x): 
            if x < 0.5: return 1 
            else: return 0
        filter_values = np.vectorize(filter_values)
        prediction = filter_values(prediction)
        return accuracy_score(y_test, prediction)

    @staticmethod
    def compute_probabilities(prediction):
        return 1-prediction