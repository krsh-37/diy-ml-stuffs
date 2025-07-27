
import numpy as np

class PCAScratch():
    def __init__(self,
                 n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # 1. center the data feature wise
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        # 2. find co-variance
        S = np.cov(X_centered, rowvar=False)
        # 3. SVD on co-variance
        U, s, Vt = np.linalg.svd(S)
        # s -> sqrt of eigenvalues/ variance explained of the PC, this is automatically sorted in desc order
        # Vt -> each row is Principal Component vector
        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = s[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(s)

    def fit_transform(self, X):
        X_cp = X.copy()
        self.fit(X_cp)
        return self.transform(X_cp)

    def transform(self, X):
        # 4. Project on the data to get reduced dimensional data
        X_centered = X - self.mean_
        return X_centered @ self.components_.T