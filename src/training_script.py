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

from src import train







def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    #   The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

if __name__ == '__main__': 
    config_parser = argparse.ArgumentParser(description='Only used as a first parser for the config_file_path')
    config_parser.add_argument('--config', type=str)
    parser = argparse.ArgumentParser() 
    parser.add_argument('prior_pfn')
    parser.add_argument('--loss_function', default='barnll')
    parser.add_argument('--min_y', default=-100.0,  type=float)
    parser.add_argument('--max_y', default=100.0, type=float)
    parser.add_argument('--num_buckets', default=100, type=int, action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL")
    parser.add_argument('--encoder', default='linear', type=str)
    parser.add_argument('--y_encoder', default='linear', type=str)
    parser.add_argument('--pos_encoder', default='sinus', type=str)
    parser.add_argument('--sequence_length', default=10, type=int)
    parser.add_argument('--warmup_epochs', default=50, type=int)
    parser.add_argument(
        '--prior_hyperparameters', 
        action=StoreDictKeyPair, 
        nargs="+", 
        metavar="KEY=VAL",
        default={},  # Default to empty dict if nothing provided
        help="Pass hyperparameters as key=value pairs, e.g. --prior_hyperparameters length_scale=0.1"
    )
    parser.add_argument('--validation_period', default=10, type=int)
    parser.add_argument('--permutation_invariant_max_eval_pos', default=None, type=int, help='Set this to an int to ')
    parser.add_argument('--permutation_invariant_sampling', default='weighted', help="Only relevant if --permutation_invariant_max_eval_pos is set.")
    parser.add_argument('--emsize', default=512, type=int) # sometimes even larger is better e.g. 1024
    parser.add_argument('--nlayers', default=6, type=int)
    parser.add_argument('--nhid', default=None, type=int) # 2*emsize is the default
    parser.add_argument('--nhead', default=4, type=int) # nhead = emsize / 64 in the original paper
    parser.add_argument('--dropout', default=.0, type=float)
    parser.add_argument('--steps_per_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--lr', '--learning_rate', default=.001, type=float) # try also .0003, .0001, go lower with lower batch size

    args, _ = _parse_args(config_parser, parser)

    if args.nhid is None:
        args.nhid = 2 * args.emsize

    prior_pfn = args.prior_pfn 

    if prior_pfn == 'gp':
        prior = gp_prior.GaussianProcessPriorGenerator() 
    else: 
        raise NotImplementedError(f'Prior == {prior_pfn}')
    
    loss_function = args.loss_function 

    num_buckets = args.num_buckets
    max_y = args.max_y
    min_y = args.min_y 
    
    if loss_function == 'ce':   
        criterion = nn.CrossEntropyLoss(reduction='none')
    elif loss_function == 'gaussnll':
        criterion = nn.GaussianNLLLoss(reduction='none', full=True)
    elif loss_function == 'mse':
        criterion = nn.MSELoss(reduction='none')
    elif loss_function == 'barnll':
        criterion = BarDistribution(borders=get_bucket_limits(num_buckets, full_range=(min_y,max_y)))
    else:
        raise NotImplementedError(f'loss_function == {loss_function}.')
    
    encoder = args.encoder 
    y_encoder = args.y_encoder 


    def get_encoder_generator(encoder):
        if encoder == 'linear':
            encoder_generator = encoders.LinearEncoder
        elif encoder == 'mlp':
            encoder_generator = encoders.MLPEncoder
        else:
            raise NotImplementedError(f'A {encoder} encoder is not valid.')
        return encoder_generator

    encoder_generator = get_encoder_generator(encoder)
    y_encoder_generator = get_encoder_generator(y_encoder)

    pos_encoder = args.pos_encoder 

    if pos_encoder == 'none':
        pos_encoder_generator = None
    elif pos_encoder == 'sinus':
        pos_encoder_generator = positional_encodings.PositionalEncoding
    elif pos_encoder == 'learned':
        pos_encoder_generator = positional_encodings.LearnedPositionalEncoding
    elif pos_encoder == 'paired_scrambled_learned':
        pos_encoder_generator = positional_encodings.PairedScrambledPositionalEncodings
    else:
        raise NotImplementedError(f'pos_encoer == {pos_encoder} is not valid.')

    permutation_invariant_max_eval_pos = args.permutation_invariant_max_eval_pos
    permutation_invariant_sampling = args.permutation_invariant_sampling

    if permutation_invariant_max_eval_pos is not None:
        if permutation_invariant_sampling == 'weighted':
            get_sampler = get_weighted_single_eval_pos_sampler
        elif permutation_invariant_sampling == 'uniform':
            get_sampler = get_uniform_single_eval_pos_sampler
        else:
            raise ValueError()
    args.__dict__['context_delimiter_generator'] = get_sampler(permutation_invariant_max_eval_pos)
    
    
    transformer_configuration = (args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    print("ARGS for `train`:", args.__dict__)

    train(prior, criterion, encoder_generator, transformer_configuration,
          y_encoder_generator=y_encoder_generator,pos_encoder_generator=pos_encoder_generator,
          **args.__dict__)
