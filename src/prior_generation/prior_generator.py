from abc import ABC, abstractmethod
from typing import Any
import prior_generation.prior_dataloader as prior_dataloader  
import torch

class PriorGenerator(ABC):
    @abstractmethod
    def get_batch(self, batch_size: int, seq_len: int, num_features: int, device: str, **hyperparameter_configuration_kwargs: Any) -> Any:
        pass 
    
    
    def get_dataloader(self, num_steps: int, fuse_x_y: bool=False, **get_batch_kwargs:Any):
        dl = prior_dataloader.get_dataloader(self.get_batch)
        out = dl(num_steps, fuse_x_y, **get_batch_kwargs) 
        return out 
    
    def get_datasets_from_prior(self, number_of_datasets, num_points_per_dataset, num_features_per_dataset, device, **hyperparameter_configuration_kwargs: Any):
        output = torch.Tensor()
        for i in range(number_of_datasets): 
            output = torch.cat(output, self.get_batch(1, num_points_per_dataset[i], num_features_per_dataset[i], device, **hyperparameter_configuration_kwargs))

