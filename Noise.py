import numpy as np

class NormalActionNoise():
    """
    A Gaussian action noise

    :param mean: (float) the mean value of the noise
    :param sigma: (float) the scale of the noise (std here)
    """

    def __init__(self, mean, sigma):
        super().__init__()
        self._mu = mean
        self._sigma = sigma

    def sample(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma)


class AnnealedGaussianProcess():
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma
    
    
    def sample(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma)
    


class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
