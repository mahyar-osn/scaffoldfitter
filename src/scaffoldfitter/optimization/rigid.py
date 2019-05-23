

class Rigid(object):
    """
    Rigid object class to handle rigid computations and optimization.
    Here a probabilistic model is used i.e. the alignment of the scaffold & data cloud is a probability density estimation
    problem. Initially, a Gaussian Mixture Model (GMM) centroid from the scaffold is fitted to the point cloud by maximizing the
    likelihood. A "coherence" constraint is imposed by reparameterization of GMM centroids with rigid parameters (i.e. scaling,
    translation, and rotation) and the closed form solution of the maximization step of the Expectation-Maximization algorithm
    is derived.
    """

    def __init__(self):
        pass

    def align(self):
        pass
