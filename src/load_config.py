import yaml
from training_util import get_uniform_single_eval_pos_sampler, get_weighted_single_eval_pos_sampler, get_cosine_schedule_with_warmup
from criterion.bar_distribution import BarDistribution, get_bucket_limits
from samplers.distribution_samplers import DistributionSampler
from samplers.distributions import ScaledBernoulli
from models import positional_encodings
from prior_generation import gp_prior, gp_lengthscale_prior
import torch.distributions as dist
import models.encoders as encoders

def load_config_from_yaml(filepath:str):
    with open(filepath, encoding='utf-8') as stream:
        try:
            
            config = yaml.safe_load(stream)
            print(yaml.dump(config, sort_keys=False, default_flow_style=False))
            
        except yaml.YAMLError as exc:
            print(exc)
    return parse_config_dict(config)
            
def parse_config_dict(config: dict):
    train_cfg = config['training_configuration']
    trans_cfg = config['transformer_configuration']
    prior_cfg = config['prior_configuration']
    crit_cfg  = config['criterion_configuration']
    defs      = config['definitions']
    
    
    #### TRAINING CONFIGURATION 
    epochs = train_cfg['epochs']
    batch_size =  train_cfg['batch_size']
    warmup_epochs = train_cfg['warmup_epochs']
    steps_per_epoch = train_cfg['steps_per_epoch']
    lr = train_cfg['lr']
    sequence_length = train_cfg['sequence_length']
    validation_context_pos = train_cfg['validation_context_pos']
    aggregate_k_gradients = train_cfg['aggregate_k_gradients']
    num_test_parameters = train_cfg["num_test_parameters"]
    num_features = defs['num_features']
    num_outputs = defs['num_outputs']
    
    scheduler = None 
    if train_cfg['scheduler'] == "cosine_scheduler":
        scheduler = get_cosine_schedule_with_warmup 
    else: 
        print("No other schedulers have been implemented yet!")
    
    #### TRANSFORMER CONFIGURATION 
    emsize = trans_cfg['emsize']
    fuse_x_y = trans_cfg['fuse_x_y']
    nlayers = trans_cfg['nlayers']
    nhead = trans_cfg['nhead']
    nhid = trans_cfg['nhid']
    dropout = trans_cfg['dropout']
    input_normalization = trans_cfg['input_normalization']
    encoder_type = trans_cfg['encoder_type']
    pos_encoder_type = trans_cfg['pos_encoder_type']
    y_encoder_type = trans_cfg['y_encoder_type']
    
    def get_prior(prior_name):
        if prior_name == "gaussian_process_prior": 
            return gp_prior.GaussianProcessPriorGenerator()
        elif prior_name == "gaussian_process_lengtscale_prior":
            return gp_lengthscale_prior.GaussianProcessHyperPriorGenerator()
        else:
            print("No such prior has been implemented ")
            
    prior = get_prior(prior_cfg["type"])
    prior_prediction = prior_cfg["prior_learning"]
            
    def get_distribution(config, param, distribution):
        if distribution == "uniform":
            low = config["samplers"][param]["low"]
            high = config["samplers"][param]["high"]
            return dist.Uniform(low=low, high=high)
        elif distribution == "scaled_bernoulli": 
            p = config["samplers"][param]["p"]
            low = config["samplers"][param]["low"]
            high = config["samplers"][param]["high"]
            return ScaledBernoulli(low, high, p)
            
            
    def get_prior_sampling_distributions(config):
        for param in config["hyperparams"]["samplers"]:
            distribution = get_distribution(config["hyperparams"], param, config["hyperparams"]["samplers"][param]['distribution'])
            config["hyperparams"]["samplers"][param]= DistributionSampler(distribution)
                
        return config["hyperparams"]
        
    if prior_prediction:
        prior_hyperparameters = get_prior_sampling_distributions(prior_cfg)
    else: 
        prior_hyperparameters = prior_cfg["hyperparams"]
        
    

            
    #### CRITERION CONFIGURATION 
    criterion = None 
    if crit_cfg['loss'] == "bar_distribution":
        num_buckets = crit_cfg['num_buckets']
        min_y = crit_cfg['min_y']
        max_y = crit_cfg['max_y']
        
        criterion = BarDistribution(borders=get_bucket_limits(num_buckets, full_range=(min_y, max_y)))
    else:
        print("No other criterion has been implemented ")
        
    def get_encoder_generator(encoder):
        if encoder == 'linear':
            encoder_generator = encoders.LinearEncoder
        elif encoder == 'mlp':
            encoder_generator = encoders.MLPEncoder
        else:
            raise NotImplementedError(f'A {encoder} encoder is not valid.')
        return encoder_generator

    encoder_generator = get_encoder_generator(encoder_type)
    y_encoder_generator = get_encoder_generator(y_encoder_type)

    if pos_encoder_type== 'sinus':
        pos_encoder_generator = positional_encodings.PositionalEncoding
    elif pos_encoder_type == 'learned':
        pos_encoder_generator = positional_encodings.LearnedPositionalEncoding
    else:
        pos_encoder_generator = positional_encodings.NoPositionalEncoding
        
    context_delimiter_sampling = train_cfg["context_delimiter_sampling"]
    context_delimiter_max_eval_pos = train_cfg["context_delimiter_max_eval_pos"]

    if context_delimiter_sampling is not None:
        if context_delimiter_sampling == 'weighted':
            get_sampler = get_weighted_single_eval_pos_sampler
        elif context_delimiter_sampling == 'uniform':
            get_sampler = get_uniform_single_eval_pos_sampler
        elif context_delimiter_sampling == 'constant_last':
            get_sampler = lambda x: sequence_length
        else:
            raise ValueError()
        
    context_delimiter_generator = get_sampler(context_delimiter_max_eval_pos)
        
    transformer_configuration = (emsize, nhead, nhid, nlayers, dropout, num_features, num_outputs, input_normalization, y_encoder_generator, sequence_length, fuse_x_y, prior_prediction, num_test_parameters) 
    training_configuration = (epochs, steps_per_epoch, batch_size, sequence_length, lr, warmup_epochs, aggregate_k_gradients, scheduler, prior_prediction, validation_context_pos)
    generators = (encoder_generator, y_encoder_generator, pos_encoder_generator)
    
    return transformer_configuration, training_configuration, criterion, generators, prior, prior_hyperparameters, context_delimiter_generator
    

