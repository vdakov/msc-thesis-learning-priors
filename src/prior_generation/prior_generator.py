from abc import ABC, abstractmethod
from typing import Any
import prior_generation.prior_dataloader as prior_dataloader  
import torch
from visualization.datasets_visualization import show_datasets
import os 
from tqdm import tqdm 
import copy
class PriorGenerator(ABC):
    @abstractmethod
    def get_batch(self, batch_size: int, seq_len: int, num_features: int, device: str, use_cache=False, **hyperparameter_configuration_kwargs: Any) -> Any:
        pass 
    
    def __init__(self):
        self.name = "Generic Prior"
        self.cache_state = {}
        self.num_cached_samples = 0

    def get_name(self):
        return self.name
    
    def get_dataloader(self, num_steps: int, fuse_x_y: bool=False, validation_context_pos=None, prior_prediction=False, **get_batch_kwargs:Any):
        dl = prior_dataloader.get_dataloader(self.get_batch)
        out = dl(num_steps, fuse_x_y, validation_context_pos, prior_prediction, **get_batch_kwargs)
        return out 
    
    def get_datasets_from_prior(self, number_of_datasets, num_points_per_dataset, num_features_per_dataset, device='cpu', **hyperparameter_configuration_kwargs: Any):
        
        x, y_noisy, y, prior_parameters = self.get_batch(number_of_datasets, num_points_per_dataset, num_features_per_dataset, device, **hyperparameter_configuration_kwargs)
            
        return x, y_noisy, y, prior_parameters
           
    def save_cache(self, path: str):
        if self.cache_state is None:
            raise ValueError("No cache to save. Run generate_cache first.")
        print(f"Saving cache to {path}...")
        torch.save(self.cache_state, path)
        
     
    def load_cache(self, path: str):
        if self.cache_state == {}:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Cache file {path} not found.")
            print(f"Loading cache from {path}...")
            
            self.cache_state = torch.load(path, map_location='cpu')
            self.num_cached_samples = self.cache_state["x"].shape[1]
            print(f"Loaded {self.num_cached_samples} samples.")
    
    def sample_cache(self, batch_size: int, device: str):
        """
        Called during training. Samples random datasets from the CPU cache
        and moves them to the GPU.
        """
        if self.cache_state is None:
            raise RuntimeError("Cache is empty. Call load_cache() or generate_cache() before sampling.")
        
        assert batch_size < self.num_cached_samples, f"Error: Requested batch {batch_size} > Cache size {self.num_cached_samples}. Sampling with replacement."
        
        indices = torch.randint(0, self.num_cached_samples, (batch_size,))
        
        x = self.cache_state["x"][:, indices, :].to(device)
        y_noisy = self.cache_state["y_noisy"][:, indices].to(device)
        y = self.cache_state["y"][:, indices].to(device)
        prior_parameters = self.cache_state["prior_parameters"][:, indices].to(device)
        
        return x, y_noisy, y, prior_parameters
 
        
    
    def cache_datasets(self, num_datasets, batch_size_per_gen, seq_len, num_features_per_dataset, device, **hyperparameter_configuration_kwargs):
        print(f"Generating cache of {num_datasets} datasets...")
        xs, y_noisys, ys, prior_parameters = [], [], [], []
        
        num_chunks = (num_datasets + batch_size_per_gen - 1) // batch_size_per_gen
        
        new_hyperparameter_configuration_kwargs = copy.deepcopy(hyperparameter_configuration_kwargs)
        new_hyperparameter_configuration_kwargs["use_cache"] = False
        
        with torch.no_grad():
            for _ in tqdm(range(num_chunks), desc='Caching Progress'):
                current_bs = min(batch_size_per_gen, num_datasets - sum(len(c) for c in xs) * batch_size_per_gen) # Logic fix for accurate count
                current_generated = sum([x.shape[1] for x in xs]) if xs else 0
                current_bs = min(batch_size_per_gen, num_datasets - current_generated)
                
                if current_bs <= 0: break

                # USE get_batch for generation as requested
                bx, by_n, by, bp = self.get_batch(
                    batch_size=current_bs, 
                    seq_len=seq_len, 
                    num_features_per_dataset=num_features_per_dataset, 
                    device=device, 
                    **new_hyperparameter_configuration_kwargs
                )
                
                xs.append(bx.cpu())
                y_noisys.append(by_n.cpu())
                ys.append(by.cpu())
                prior_parameters.append(bp.cpu())
                
        self.cache_state = {
            "x": torch.cat(xs, dim=1),
            "y_noisy": torch.cat(y_noisys, dim=1),
            "y": torch.cat(ys, dim=1),
            "prior_parameters": torch.cat(prior_parameters, dim=1),
            "settings": {"seq_len": seq_len, "num_features_per_dataset": num_features_per_dataset}
        }
        self.num_cached_samples = self.cache_state["x"].shape[1]
        print(f"Cache generated. Total samples: {self.num_cached_samples}")
        
        
    def visualize_datasets(self, number_of_datasets, num_points_per_dataset, num_features_per_dataset, device='cpu', **hyperparameter_configuration_kwargs: Any):
        datasets = self.get_datasets_from_prior(number_of_datasets, num_points_per_dataset, num_features_per_dataset, **hyperparameter_configuration_kwargs)
        x, y_noisy, y, _ = datasets 
        x, y_noisy, y,= x.detach().numpy() , y.detach().numpy() , y_noisy.detach().numpy()
        assert x.shape[2] == 1, "Only one-dimensional x-datasets possible!"
        show_datasets(x, y, y_noisy, "Datasets from prior: " + self.get_name())
        
    
        

