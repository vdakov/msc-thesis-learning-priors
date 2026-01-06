import torch 

def compare_pfn_to_model(prior_generator, prior_hyperparameters, num_datasets, num_points_in_dataset, pfn_model, dataset_size, context_size, device):
    train_X, train_Y, y_target = prior_generator.get_datasets_from_prior(num_datasets, num_points_in_dataset, 1, **prior_hyperparameters)
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    y_target = y_target.to(device)
    
    for index in range(num_datasets):
        train_x = train_X[:context_size, index, :]
        train_y = train_Y[:context_size, index]
        test_x = train_X[:, index, :]
        
        with torch.no_grad():
     
