from torch.utils.data import DataLoader, IterableDataset
from typing import Callable, Tuple, Any, Dict, Iterator, Union, Optional
import torch

def set_locals_in_self(locals):
    self = locals['self']
    for var_name, val in locals.items():
        if var_name != 'self': setattr(self, var_name, val)

PriorBatchMethod = Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

def get_dataloader(get_prior_batch_method: PriorBatchMethod) -> Callable[..., DataLoader]:
    '''
    Returns a dataloader, retrieving samples from a given prior. Local state 
    is saved as "self"'s arguments. 
    '''

    class PriorDataset(IterableDataset):
        def __init__(self, num_steps: int, fuse_x_y: bool = False, **get_batch_kwargs: Any):
            set_locals_in_self(locals())
            self.num_steps = num_steps
            self.num_features = get_batch_kwargs.get('num_features')
            self.num_outputs = get_batch_kwargs.get('num_outputs')
            self.get_batch_kwargs = get_batch_kwargs
            self.fuse_x_y = fuse_x_y
            print('Dataset.__dict__', self.__dict__)
        
        def __iter__(self): 
            x, y, target_y = get_prior_batch_method(**self.get_batch_kwargs)
            if self.fuse_x_y:
                yield torch.cat([x, torch.cat([torch.zeros_like(y[:1]), y[:-1]], 0).unsqueeze(-1).float()],
                                 -1), target_y
            else:
                yield (x, y), target_y

        def __len__(self): 
            return self.num_steps 


    class PriorDataLoader(DataLoader):
        def __init__(self, num_steps: int, fuse_x_y: bool = False, **get_batch_kwargs: Union[int, bool, Any]):
            set_locals_in_self(locals())
            self.num_features = get_batch_kwargs.get('num_features')
            self.num_outputs = get_batch_kwargs.get('num_outputs')
            dataset = PriorDataset(num_steps, fuse_x_y, **get_batch_kwargs)
            super().__init__(dataset, batch_size=None)
            print('DataLoader.__dict__', self.__dict__)

    return PriorDataLoader


