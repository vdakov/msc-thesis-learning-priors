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



class GaussianProcessPriorGenerator(PriorGenerator):
    KERNEL_MAP = {
        "rbf": gpytorch.kernels.RBFKernel,
        "matern": gpytorch.kernels.MaternKernel,
        "periodic": gpytorch.kernels.PeriodicKernel,
        "cosine": gpytorch.kernels.CosineKernel,
        "linear": gpytorch.kernels.LinearKernel,
        "spectral": gpytorch.kernels.SpectralMixtureKernel,
    }
    
    def __init__(self):
        super()
        self.name = "Gaussian Process Prior"

    def _get_kernel(self, kernel_name: str, **kwargs) -> gpytorch.kernels.ScaleKernel: 
        kernel_cls = self.KERNEL_MAP.get(kernel_name)
        if not kernel_cls:
            raise ValueError(f'Kernel name {kernel_name} not supported')
        init_kwargs = {}
        if kernel_name == 'matern':
            init_kwargs['nu'] = kwargs['nu']

        base_kernel = kernel_cls(**init_kwargs)

        return gpytorch.kernels.ScaleKernel(base_kernel)

    def get_batch(self, batch_size:int, seq_len:int, num_features: int, device: str, **hyperparameter_configuration_kwargs: Any): 
        x = torch.rand(batch_size, seq_len, num_features, device=device)
        kernel_name = hyperparameter_configuration_kwargs.get('kernel_name', 'rbf') #type ignore 
        length_scale =  hyperparameter_configuration_kwargs.get('length_scale', 1) #type ignore 
        output_scale = hyperparameter_configuration_kwargs.get('output_scale', 1) #type ignore 
        noise_std = hyperparameter_configuration_kwargs.get('noise_std', 0.1) #type ignore 

        kernel = self._get_kernel(kernel_name, **hyperparameter_configuration_kwargs) #type ignore 
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

        return x, y, y

    
    
        
