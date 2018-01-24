from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.core.utils.data_structures import InstanceWindow
from skmultiflow.core.utils.utils import get_dimensions


class BatchClassifier:

    def __init__(self, window_size=100, max_models=10):
        self.H = []
        self.h = None
        # TODO
        self.window_size = window_size
        self.window = InstanceWindow(max_size=window_size, dtype=float)
        self.max_models = max_models

    def partial_fit(self, X, y=None, classes=None):
        # TODO
        r, c = get_dimensions(X)
        if self.window is None:
            self.window = InstanceWindow(max_size=self.window_size, dtype=float)
        # models = []
        modeles = 0
        if not self.H:
            # Slice pretraining set
            debut = 0
            fin = self.window_size
            while (modeles < self.max_models):
                X_batch = X[debut:fin, :]
                y_batch = y[debut:fin]
                debut += self.window_size
                fin += self.window_size
                self.h = DecisionTreeClassifier()
                self.h.fit(X_batch, y_batch)
                self.H.append(self.h)  # <-- and append it to the ensemble
                modeles += 1
        else:
            for i in range(r):
                self.window.add_element(np.asarray([X[i]]), np.asarray([[y[i]]]))
            for model in range(modeles):
                self.h = DecisionTreeClassifier()
                self.h.fit(self.window.get_attributes_matrix(), self.window.get_targets_matrix())
                self.H.append(self.h)  # <-- and append it to the ensemble
        return self

    def predict(self, X):
        # TODO
        N, _ = X.shape
        predictions = []
        y = []
        for h in self.H:
            y.append(h.predict(X))
        for i in range(N):
            votes = Counter([j[i] for j in y])
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append(0)
            else:
                predictions.append(max(votes, key=votes.get))
        return predictions
