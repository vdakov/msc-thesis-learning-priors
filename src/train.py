import random 
import argparse
import torch.nn as nn 
from tqdm import tqdm
import yaml
from criterion.bar_distribution import BarDistribution, get_bucket_limits
import time
import torch
from prior_generation import gp_prior 
import models.encoders as encoders
import models.positional_encodings as positional_encodings 
from models.transformer import TransformerModel
from models.prior_transformer import PriorTransformerModel
import math 
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import csv 
import os
import time
import numpy as np
import random


def load_transformer(transformer_configuration, generators):
    emsize, nhead, nhid, nlayers, dropout, num_features, n_out, input_normalization, y_encoder_generator, sequence_length, fuse_x_y, prior_prediction, num_test_parameters  = transformer_configuration
    encoder_generator, y_encoder_generator, pos_encoder_generator = generators
    encoder = encoder_generator(num_features + 1 if fuse_x_y else num_features,emsize)
    y_encoder = y_encoder_generator(1, emsize)
    pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, sequence_length * 2)
    if prior_prediction:

        model = PriorTransformerModel(encoder, n_out, emsize, nhead, nhid, nlayers, dropout, num_test_parameters, num_features,
                                y_encoder=y_encoder, input_normalization=input_normalization,
                                pos_encoder=pos_encoder)
    else:
        model = TransformerModel(encoder, n_out, emsize, nhead, nhid, nlayers, dropout,
                                y_encoder=y_encoder, input_normalization=input_normalization,
                                pos_encoder=pos_encoder)
    return model 

def set_random_seeds(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def train(prior_dataloader, criterion, transformer_configuration, generators, training_configuration,
          prior_hyperparameters, load_path=None, context_delimiter_generator=None, device='cuda:0', 
          save_folder=None, experiment_name=None, seed=42, **kwargs):

    epochs, steps_per_epoch, batch_size, sequence_length, lr, warmup_epochs, aggregate_k_gradients, scheduler, prior_prediction, validation_context_pos = training_configuration
    dataloader = prior_dataloader.get_dataloader(num_steps=steps_per_epoch, validation_context_pos=validation_context_pos, batch_size=batch_size, seq_len=sequence_length, device=device,  prior_prediction=prior_prediction, **prior_hyperparameters)
    model = load_transformer(transformer_configuration, generators)
    n_out = dataloader.num_outputs
    
    set_random_seeds(seed)
    
    print(f'Using {device} device')
    print('Total Number of Datasets:', batch_size * epochs * steps_per_epoch)
    
    if load_path is not None:
        return load_checkpoint(load_path, transformer_configuration, generators, criterion, device=device)
    
    model.criterion = criterion
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = scheduler(optimizer, warmup_epochs, epochs)

    def train_one_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        
        assert len(dataloader) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        for batch, (data, targets, prior_parameters) in enumerate(dataloader):
            # if not prior_prediction:
            context_delimiter = context_delimiter_generator() if callable(context_delimiter_generator) else context_delimiter_generator
            # else:
            #     context_delimiter = len(targets)

            
            if prior_prediction:
                for _, pp in enumerate(prior_parameters):
                    pp = pp.unsqueeze(0)
                    targets = torch.cat((targets, pp))
                    
            
            output = model(tuple(e.to(device) for e in data) if isinstance(data, (tuple, list)) else data.to(device), context_pos=context_delimiter)
            
            if context_delimiter is not None:
                targets = targets[context_delimiter:]

            losses = criterion(output.reshape(-1, n_out), targets.to(device).flatten())
            
            losses = losses.view(*output.shape[0:2]).squeeze(-1)
            loss = losses.mean()
            loss.backward()
            if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item()
            if prior_prediction: 
                
                total_positional_losses += losses.mean(1).cpu().detach() if context_delimiter is None else \
                    nn.functional.one_hot(torch.tensor(context_delimiter), sequence_length + len(prior_parameters))*loss.cpu().detach()

                total_positional_losses_recorded += torch.ones(sequence_length) if context_delimiter is None else \
                    nn.functional.one_hot(torch.tensor(context_delimiter), sequence_length + len(prior_parameters))
            else: 
                                
                total_positional_losses += losses.mean(1).cpu().detach() if context_delimiter is None else \
                    nn.functional.one_hot(torch.tensor(context_delimiter), sequence_length)*loss.cpu().detach()

                total_positional_losses_recorded += torch.ones(sequence_length) if context_delimiter is None else \
                    nn.functional.one_hot(torch.tensor(context_delimiter), sequence_length)
        
            
        return total_loss / steps_per_epoch, (total_positional_losses / total_positional_losses_recorded).tolist()

    best_val_loss = float("inf")
    losses, positional_losses, val_losses = [], [], []
    progress_bar = tqdm(range(1, epochs + 1))
    for epoch in progress_bar:
        loss, positional_loss = train_one_epoch() 
        losses.append(loss)
        positional_losses.append(positional_loss)
        if hasattr(dataloader, 'validate'):
            with torch.no_grad():
                val_score = dataloader.validate(model, criterion, device)
                val_losses.append(val_score)
                best_val_loss = min(val_score, best_val_loss)
        else:
            val_score = None

        
        desc = f'loss {loss:5.2f} | pos loss {','.join([f'{l:5.2f}' for l in positional_loss])}, lr {scheduler.get_last_lr()[0]}' + (f' val score {val_score}' if val_score is not None else '')
        progress_bar.set_description(desc)
        scheduler.step()
        
    if save_folder is not None:
        experiment_name = experiment_name or f'model-{time.time()}'
        exp_dir = os.path.join(save_folder, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)

        checkpoint_path = os.path.join(exp_dir, 'checkpoint.pt')
        
        # Save everything in one go
        save_checkpoint(
            model, optimizer, scheduler, epoch, 
            losses, positional_losses, val_losses, 
            config=training_configuration, # handy for debugging later
            path=checkpoint_path
        )
        print(f"Full experiment checkpoint saved to {checkpoint_path}")

    
        
    
    return model.to('cpu'), losses, positional_losses, val_losses


def save_checkpoint(model, optimizer, scheduler, epoch, losses, positional_losses, val_losses, config, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
        'positional_losses': positional_losses,
        'val_losses': val_losses,
        'config': config, # Store the YAML config here too!
    }
    torch.save(checkpoint, path)
    
def load_checkpoint(path, transformer_configuration, generators, criterion, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model = load_transformer(transformer_configuration, generators)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.criterion = criterion 
    
    return model, checkpoint['losses'], checkpoint['positional_losses'], checkpoint['val_losses']