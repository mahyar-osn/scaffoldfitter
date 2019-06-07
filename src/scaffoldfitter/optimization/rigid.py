import numpy as np


class Rigid(object):
    """
    Rigid object class to handle rigid computations and optimization.
    Here a probabilistic model is used i.e. the alignment of the scaffold & data cloud is a probability density estimation
    problem. Initially, a Gaussian Mixture Model (GMM) centroid from the scaffold is fitted to the point cloud by maximizing the
    likelihood. A "coherence" constraint is imposed by re-parametrisation of GMM centroids with rigid parameters (i.e. scaling,
    translation, and rotation) and the closed form solution of the maximization step of the Expectation-Maximization algorithm
    is derived.
    """

    def __init__(self, X, Y, R=None, t=None, s=None, sigma2=None, max_iter=100, tolerance=0.001, w=0):
        self.X = X
        self.Y = Y
        self.TY = Y
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.R = np.eye(self.D) if R is None else R
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
        self.s = 1 if s is None else s
        self.sigma2 = sigma2
        self.iteration = 0
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.w = w
        self.q = 0
        self.err = 0

    def register(self, callback):
        self.initialize()
        while self.iteration < self.max_iter and self.err > self.tolerance:
            self.iterate()
            if callback:
                callback(iteration=self.iteration, error=self.err, X=self.X, Y=self.TY)
        return self.TY, self.R, np.atleast_2d(self.t), self.s

    def iterate(self):
        self.expectation()
        self.maximize()
        self.iteration += 1

    def maximize(self):
        self.update_transformation_matrix()
        self.transform_data()
        self.update_variance()

    def update_transformation_matrix(self):
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
        muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)
        self.XX = self.X - np.tile(muX, (self.N, 1))
        YY = self.Y - np.tile(muY, (self.M, 1))

        self.A = np.dot(np.transpose(self.XX), np.transpose(self.P))
        self.A = np.dot(self.A, YY)

        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D,))
        C[self.D - 1] = np.linalg.det(np.dot(U, V))

        self.R = np.dot(np.dot(U, np.diag(C)), V)
        self.YPY = np.dot(np.transpose(self.P1), np.sum(np.multiply(YY, YY), axis=1))
        self.s = np.trace(np.dot(np.transpose(self.A), self.R)) / self.YPY
        self.t = np.transpose(muX) - self.s * np.dot(self.R, np.transpose(muY))

    def transform_data(self, Y=None):
        if Y is None:
            self.TY = self.s * np.dot(self.Y, np.transpose(self.R)) + np.tile(np.transpose(self.t), (self.M, 1))
            return
        else:
            return self.s * np.dot(Y, np.transpose(self.R)) + np.tile(np.transpose(self.t), (Y.shape[0], 1))

    def update_variance(self):
        qprev = self.q
        trAR = np.trace(np.dot(self.A, np.transpose(self.R)))
        xPx = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.XX, self.XX), axis=1))
        self.q = (xPx - 2 * self.s * trAR + self.s * self.s * self.YPY) / (
                    2 * self.sigma2) + self.D * self.Np / 2 * np.log(self.sigma2)
        self.err = np.abs(self.q - qprev)

        self.sigma2 = (xPx - self.s * trAR) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def initialize(self):
        self.TY = self.s * np.dot(self.Y, np.transpose(self.R)) + np.repeat(self.t, self.M, axis=0)
        if not self.sigma2:
            XX = np.reshape(self.X, (1, self.N, self.D))
            YY = np.reshape(self.TY, (self.M, 1, self.D))
            XX = np.tile(XX, (self.M, 1, 1))
            YY = np.tile(YY, (1, self.N, 1))
            diff = XX - YY
            err = np.multiply(diff, diff)
            self.sigma2 = np.sum(err) / (self.D * self.M * self.N)

        self.err = self.tolerance + 1
        self.q = -self.err - self.N * self.D / 2 * np.log(self.sigma2)

    def expectation(self):
        P = np.zeros((self.M, self.N))

        for i in range(0, self.M):
            diff = self.X - np.tile(self.TY[i, :], (self.N, 1))
            diff = np.multiply(diff, diff)
            P[i, :] = P[i, :] + np.sum(diff, axis=1)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        P = np.exp(-P / (2 * self.sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)

