import torch
from torch.distributions import Distribution

class DistributionSampler:
    def __init__(self, distribution: Distribution):
        """
        :param distribution: An instantiated torch.distributions object 
                             (e.g., Uniform(0.1, 3.0), Gamma(2.0, 2.0))
        """
        self.distribution = distribution

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Samples from the distribution with the given batch size.
        Returns a tensor of shape (batch_size,)
        """
        # PyTorch distributions expect a tuple/torch.Size for sampling
        return self.distribution.sample(torch.Size([batch_size]))
        
