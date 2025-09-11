import numpy as np
from scipy.stats import gaussian_kde

class BaseMassDistribution:
    def sample(self) -> np.ndarray:
        raise NotImplementedError


class UniformMassDistribution(BaseMassDistribution):
    def __init__(self, low: float = 0.5, high: float = 1.5, seed: int = None):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=3)


class NormalMassDistribution(BaseMassDistribution):
    def __init__(self, mu, sigma, seed: int = None):
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> np.ndarray:
        scale =  np.clip(
                np.random.normal(self.mu, self.sigma, size=3),
                a_min=0.01,  # Avoid non-positive masses
                a_max=None)
        return  scale 
    
    
class DegenerateMassDistribution(BaseMassDistribution):
    def sample(self) -> np.ndarray:
        return np.ones(3)

