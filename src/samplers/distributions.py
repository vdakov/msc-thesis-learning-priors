from torch.distributions import Distribution, Bernoulli
import torch 

class ScaledBernoulli(Distribution):
    def __init__(self, low, high, prob=0.5):
        """
        A Bernoulli-like distribution that returns 'low' or 'high' 
        instead of 0 or 1.
        
        :param low: The value to return when Bernoulli samples 0
        :param high: The value to return when Bernoulli samples 1
        :param prob: Probability of sampling 'high' (default 0.5)
        """
        super().__init__()
        self.low = low
        self.high = high
        self.prob = prob
        self.base_dist = Bernoulli(torch.tensor([prob]))

    def sample(self, sample_shape=torch.Size()):
        # 1. Sample 0s and 1s
        # We must squeeze the last dim because Bernoulli adds an extra dimension
        mask = self.base_dist.sample(sample_shape).squeeze(-1)
        
        # 2. Scale and Shift: (mask * (high - low)) + low
        return mask * (self.high - self.low) + self.low