import gpytorch
from typing import Any, Tuple 
import torch
from prior_generation.prior_generator import PriorGenerator
import prior_generation.prior_dataloader as prior_dataloader 

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, mean_module, covariance_module, likelihood):
        super(ExactGPModel, self).__init__(None, None, likelihood)
        self.mean_module = mean_module
        self.covar_module = covariance_module

    def forward(self, x):
        mean_x: torch.Tensor = self.mean_module(x) # type: ignore
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class BayesianLinearRegressionHyperPrior(PriorGenerator):

    def __init__(self):
        super()
        self.name = "Bayesian Linear Regression HyperPrior"

    def get_batch(self, batch_size:int, seq_len:int, num_features: int, device: str, **hyperparameter_configuration_kwargs: Any): 
        x = torch.randn(batch_size, seq_len, num_features, device=device)
        mean_sampling = hyperparameter_configuration_kwargs["samplers"]["mean"] #type ignore 
        std_prior_sampling = hyperparameter_configuration_kwargs["samplers"]["std"] #type ignore 
        noise_std = hyperparameter_configuration_kwargs.get('noise_std', 0.1) #type ignore 
        
        length_scale =  length_scale_sampling.sample(batch_size).to(device)
        length_scale = length_scale.view(batch_size, 1, 1)

        kernel = self._get_kernel(kernel_name, batch_size, **hyperparameter_configuration_kwargs) #type ignore 
        kernel.output_scale = output_scale #type ignore 

        kernel.base_kernel.lengthscale = length_scale #type ignore 
        kernel = kernel.to(device)
        
        covar_module = kernel(x)
        mean_module = torch.zeros(batch_size, seq_len, device=device)
       
        dist = gpytorch.distributions.MultivariateNormal(mean_module, covar_module)
        y = dist.rsample()
        y_noisy = y + torch.multiply(torch.randn_like(y), noise_std)
        x = x.transpose(0, 1) 
        y = y.transpose(0, 1)
        y_noisy = y_noisy.transpose(0, 1)

        return x, y_noisy, y, length_scale.view(1, batch_size)

    
    
        
