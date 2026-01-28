import argparse
import train
import load_config
import json 
import torch
import sys 
import time 

def get_args():
    config_parser = argparse.ArgumentParser(
        prog="Prior-Learning PFNs Training Loop",
        description="Input either a config file path or the arguments you want to load",
    )

    config_parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    definitions = config_parser.add_argument_group("Definitions")
    definitions.add_argument("--num_features", type=int, default=1)
    definitions.add_argument("--num_outputs", type=int, default=100)
    definitions.add_argument("--sequence_length", type=int, default=25)
    definitions.add_argument("--max_eval_pos", type=int, default=25)

    training_configuration = config_parser.add_argument_group("Training Configuration")
    training_configuration.add_argument("--epochs", type=int, default=500)
    training_configuration.add_argument("--batch_size", type=int, default=256)
    training_configuration.add_argument(
        "--warmup_epochs", type=int, default=25
    )  
    training_configuration.add_argument(
        "--steps_per_epoch", type=int, default=10
    ) 
    training_configuration.add_argument(
        "--validation_context_pos", type=int, default=None
    )
    training_configuration.add_argument(
        "--lr", type=float, default=0.0001
    )  
    training_configuration.add_argument(
        "--scheduler", type=str, default="cosine_scheduler"
    )
    training_configuration.add_argument("--aggregate_k_gradients", type=int, default=1)
    training_configuration.add_argument(
        "--context_delimiter_sampling", type=str, default="constant_last"
    )  
    training_configuration.add_argument(
        "--context_delimiter_max_eval_pos", type=int, default=None
    )  
    training_configuration.add_argument("--num_test_parameters", type=int, default=1)
    transformer_configration = config_parser.add_argument_group(
        "Transformer Configuration"
    )
    transformer_configration.add_argument("--emsize", type=int, default=512)
    
    transformer_configration.add_argument("--fuse_x_y", action="store_true")
    transformer_configration.add_argument("--nlayers", type=int, default=6)
    transformer_configration.add_argument("--nhead", type=int, default=4)
    transformer_configration.add_argument("--nhid", type=int, default=1024)
    transformer_configration.add_argument("--dropout", type=float, default=0.2)
    transformer_configration.add_argument(
        "--input_normalization", action="store_true"
    )  
    transformer_configration.add_argument("--encoder_type", type=str, default="mlp")
    transformer_configration.add_argument(
        "--pos_encoder_type", type=str, default="none"
    )
    transformer_configration.add_argument(
        "--y_encoder_type", type=str, default="linear"
    )

    prior_configuration = config_parser.add_argument_group("Prior Configuration")
    
    prior_configuration.add_argument("--prior_learning", action="store_true")
    prior_configuration.add_argument(
        "--prior_type", type=str, default="gaussian_process_prior"
    )  
    prior_configuration.add_argument(
        "--prior_hyperparams",
        type=str,
        default=None,
        help="JSON string to override the entire hyperparams dict",
    )
    prior_configuration.add_argument(
        "--use_cache", action="store_true"
    ) 
    prior_configuration.add_argument("--cache_path", type=str, default="")

    criterion_configuration = config_parser.add_argument_group(
        "Criterion Configuration"
    )
    criterion_configuration.add_argument("--loss", type=str, default="bar_distribution")
    criterion_configuration.add_argument(
        "--min_y", type=float, default=-5.0
    )  # Changed str to float
    criterion_configuration.add_argument(
        "--max_y", type=float, default=5.0
    )  # Changed str to float

    # --- Metadata ---
    config_parser.add_argument("--save_folder", type=str, default="../results")
    config_parser.add_argument("--load_path", type=str, default=None)
    config_parser.add_argument("--experiment_name", type=str, default=f'experiment-{time.time()}')

    return config_parser

def make_args_into_dict(args):
    hyperparams = {}
    if args.prior_hyperparams:
        try:
            hyperparams = json.loads(args.prior_hyperparams)
        except json.JSONDecodeError as e:
            print(f"Error parsing --prior_hyperparams: {e}")
            sys.exit(1)
            
    assert args.sequence_length >= args.validation_context_pos, "The sequence is large enough for the model evaluation context"
    assert args.sequence_length >= args.context_delimiter_max_eval_pos, (
        "The sequence is large enough for the model evaluation context"
    )
    assert args.min_y < args.max_y, "Loss range is not valid"
    
    
    args_dict = {}
    training_configuration = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "warmup_epochs": args.warmup_epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "validation_context_pos": args.validation_context_pos,
        "sequence_length": args.sequence_length,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "aggregate_k_gradients": args.aggregate_k_gradients,
        "context_delimiter_sampling": args.context_delimiter_sampling,
        "context_delimiter_max_eval_pos": args.context_delimiter_max_eval_pos,
        "num_test_parameters": args.num_test_parameters,
    }
    
    transformer_configuration = {
        "emsize": args.emsize,
        "fuse_x_y": args.fuse_x_y,
        "nlayers": args.nlayers,
        "nhead": args.nhead,
        "nhid": args.nhid,
        "dropout": args.dropout,
        "input_normalization": args.input_normalization,
        "encoder_type": args.encoder_type,
        "pos_encoder_type": args.pos_encoder_type,
        "y_encoder_type": args.y_encoder_type,
        "num_features": args.num_features,
        "num_outputs": args.num_outputs,
    }
    prior_configuration = {
        "prior_learning": args.prior_learning,
        "type": args.prior_type,
        "hyperparams": hyperparams,
        "use_cache": args.use_cache,
        "cache_path": args.cache_path,
    }
    criterion_configuration = {
        "loss": args.loss,
        "min_y": args.min_y,
        "max_y": args.max_y,
        "num_buckets": args.num_outputs,
    }
    
    args_dict['training_configuration'] = training_configuration
    args_dict["transformer_configuration"] = transformer_configuration
    args_dict["prior_configuration"] = prior_configuration
    args_dict["criterion_configuration"] = criterion_configuration
    
    return args_dict

if __name__ == '__main__': 
    parser = get_args()
    args = parser.parse_args()
    config = args.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if config: 
        transformer_configuration, training_configuration, criterion, generators, prior, prior_hyperparameters, context_delimiter_generator = load_config.load_config_from_yaml(config)
    else: 
        args_dict = make_args_into_dict(args)
        transformer_configuration, training_configuration, criterion, generators, prior, prior_hyperparameters, context_delimiter_generator = load_config.parse_config_dict(config)
    
    model, losses, positional_losses, val_losses = train.train(
        prior_dataloader=prior,
        criterion=criterion,  # Passing the wrapper
        transformer_configuration=transformer_configuration,
        generators=generators,
        training_configuration=training_configuration,
        prior_hyperparameters=prior_hyperparameters,
        context_delimiter_generator=context_delimiter_generator,
        save_folder=args.save_folder,
        load_path=args.load_path,
        experiment_name=args.experiment_name,
        device=device,
    )
        
        
    