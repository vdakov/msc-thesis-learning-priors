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
import math 
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

# copied from huggingface
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def load_transformer(transormer_configuration, generators, decoder):
    emsize, nhead, nhid, nlayers, dropout, n_in, n_out, input_normalization, y_encoder_generator, sequence_length, fuse_x_y  = transformer_configuration
    encoder_generator, y_encoder_generator, pos_encoder_generator = generators
    encoder = encoder_generator(n_in + 1 if fuse_x_y else n_in,emsize)
    y_encoder = y_encoder_generator(1, emsize)
    pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, sequence_length * 2)
    model = TransformerModel(encoder, n_out, emsize, nhead, nhid, nlayers, dropout,
                             y_encoder=y_encoder, input_normalization=input_normalization,
                             pos_encoder=pos_encoder,
                             decoder=decoder
                             )
    return model 
    

def train(prior_dataloader, criterion, encoder_generator, transformer_configuration, generators, decoder, training_configuration,
          prior_hyperparameters, load_path=None, context_delimiter_generator=None, device='cuda:0',
          aggregate_k_gradients=1, verbose=True, save_path = None, **kwargs):

    
    device = device if torch.cuda.is_available() else "cpu:0"
    print(f'Using {device} device')
    epochs, steps_per_epoch, batch_size, sequence_length, lr, warmup_epochs, validation_period, aggregate_k_gradients, scheduler = training_configuration
    dataloader = prior_dataloader.get_dataloader(num_steps=steps_per_epoch, batch_size=batch_size, seq_len=sequence_length, **prior_hyperparameters)
    model = load_transformer(transformer_configuration, generators, decoder)
    
    model.criterion = criterion
    if load_path is not None:
        model.load_state_dict(load_path)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = scheduler(optimizer, warmup_epochs, epochs)

    def train_one_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        
        assert len(dataloader) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        for batch, (data, targets) in enumerate(dataloader):
            context_delimiter = context_delimiter_generator() if callable(context_delimiter_generator) else context_delimiter_generator
    
            output = model(tuple(e.to(device) for e in data) if isinstance(data, (tuple, list)) else data.to(device)) 
            
            if context_delimiter is not None:
                targets = targets[context_delimiter:]

            losses = losses.view(*output.shape[0:2]).squeeze(-1)
            loss = losses.mean()
            loss.backward()
            if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_positional_losses += losses.mean(1).cpu().detach() if context_delimiter is None else \
                nn.functional.one_hot(torch.tensor(context_delimiter), sequence_length)*loss.cpu().detach()

            total_positional_losses_recorded += torch.ones(sequence_length) if context_delimiter is None else \
                nn.functional.one_hot(torch.tensor(sequence_length), sequence_length)

            
        return total_loss / steps_per_epoch, (
                    total_positional_losses / total_positional_losses_recorded).tolist()

    best_val_loss = float("inf")
    losses, positional_losses = [], []
    for epoch in tqdm(range(1, epochs + 1)):
        loss, positional_loss = train_one_epoch() 
        losses.append(loss)
        positional_losses.append(loss)
        if hasattr(dataloader, 'validate') and epoch % validation_period == 0:
            with torch.no_grad():
                val_score = dataloader.validate(model)
                best_val_loss = min(val_score, best_val_loss)
                if val_score == best_val_loss and save_path is not None:
                    model._save_to_state_dict(save_path)
        else:
            val_score = None

        if verbose:
            print('-' * 89)
            print(
                f'| end of epoch {epoch:3d} | loss {loss:5.2f} | '
                f"pos loss {','.join([f'{l:5.2f}' for l in positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                (f'val score {val_score}' if val_score is not None else ''))
            print('-' * 89)

        scheduler.step()
        
    
    return losses, positional_losses, model.to('cpu')
