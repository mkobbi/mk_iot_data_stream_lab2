from numpy import *

from sklearn.tree import DecisionTreeClassifier

class BatchClassifier:

    def __init__(self, window_size=100, max_models=10):
        self.H = []
        self.h = None
        # TODO

    def partial_fit(self, X, y=None, classes=None):
        # TODO 
        #if not initialized ...
            # Setup 
        # N.B.: The 'classes' option is not important for this classifier
        # HINT: You can build a decision tree model on a set of data like this:
        #       h = DecisionTreeClassifier()
        #       h.fit(X_batch,y_batch)
        #       self.H.append(h) # <-- and append it to the ensemble

        return self

    def predict(self, X):
        # TODO 
        N,D = X.shape
        # You also need to change this line to return your prediction instead of 0s:
        return zeros(N) 

