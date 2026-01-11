from abc import ABC, abstractmethod
from typing import Any
import prior_generation.prior_dataloader as prior_dataloader  
import torch
from visualization.datasets_visualization import show_datasets

class PriorGenerator(ABC):
    @abstractmethod
    def get_batch(self, batch_size: int, seq_len: int, num_features: int, device: str, **hyperparameter_configuration_kwargs: Any) -> Any:
        pass 
    
    def __init__(self):
        self.name = "Generic Prior"

    def get_name(self):
        return self.name
    
    def get_dataloader(self, num_steps: int, fuse_x_y: bool=False, validation_context_pos=None, prior_prediction=False, **get_batch_kwargs:Any):
        dl = prior_dataloader.get_dataloader(self.get_batch)
        out = dl(num_steps, fuse_x_y, validation_context_pos, prior_prediction, **get_batch_kwargs)
        return out 
    
    def get_datasets_from_prior(self, number_of_datasets, num_points_per_dataset, num_features_per_dataset, device='cpu', **hyperparameter_configuration_kwargs: Any):
        
        x, y_noisy, y, prior_parameters = self.get_batch(number_of_datasets, num_points_per_dataset, num_features_per_dataset, device, **hyperparameter_configuration_kwargs)
            
        return x, y_noisy, y, prior_parameters
    
    def visualize_datasets(self, number_of_datasets, num_points_per_dataset, num_features_per_dataset, device='cpu', **hyperparameter_configuration_kwargs: Any):
        datasets = self.get_datasets_from_prior(number_of_datasets, num_points_per_dataset, num_features_per_dataset, **hyperparameter_configuration_kwargs)
        x, y_noisy, y, _ = datasets 
        x, y_noisy, y,= x.detach().numpy() , y.detach().numpy() , y_noisy.detach().numpy()
        assert x.shape[2] == 1, "Only one-dimensional x-datasets possible!"
        show_datasets(x, y, y_noisy, "Datasets from prior: " + self.get_name())
        
        

