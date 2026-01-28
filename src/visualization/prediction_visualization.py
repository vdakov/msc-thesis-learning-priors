import torch 
import matplotlib.pyplot as plt 
import numpy as np
from model_evaluation.evaluate_with_context import evaluate_parameter_distributions_on_model


def show_vanilla_pfn_predictions(model, train_X, train_Y, num_datasets, num_training_points, device, show_plot=True, save_path=None):

    model = model.to(device)
    # Set up grid for subplots
    _, axes = plt.subplots(3, 3, figsize=(15, 8)) 
    axes = axes.flatten()

    for batch_index in range(num_datasets):
        ax = axes[batch_index] 
        train_x = train_X[:num_training_points, batch_index, :]
        train_y = train_Y[:num_training_points, batch_index]
        test_x = train_X[:, batch_index, :]
        with torch.no_grad():
            logits = model((torch.cat((train_x, test_x)), torch.cat((train_y, torch.zeros(len(test_x), device=device)))), context_pos=num_training_points-1)

            pred_means = model.criterion.mean(logits)
            pred_confs = model.criterion.quantile(logits)
            pred_means = pred_means[-len(test_x):]

            pred_confs = pred_confs[-len(test_x):]
            # Plot scatter points for training data
            ax.scatter(train_x[..., 0].cpu().numpy(), train_y.cpu().numpy(), label="Seen Data")
            ax.scatter(train_X[num_training_points:, batch_index, :].cpu().numpy(), train_Y[num_training_points:, batch_index].cpu().numpy(), label="Unseen Data")

        # Plot model predictions
        order_test_x = test_x[:, 0].cpu().argsort()
        ax.plot(
            test_x[order_test_x, 0].cpu().numpy(),
            pred_means[order_test_x].cpu().numpy(),
            color='green',
            label='pfn'
        )
        ax.fill_between(
            test_x[order_test_x, 0].cpu().numpy().flatten(),         # Flatten X
            pred_confs[order_test_x][..., 0].cpu().numpy().flatten(), # Flatten Y1 (Lower Bound)
            pred_confs[order_test_x][..., 1].cpu().numpy().flatten(), # Flatten Y2 (Upper Bound)
            alpha=.1,
            color='green'
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    if show_plot:
        plt.show()
        
def save_vanilla_pfn_predictions(model, prior, prior_hyperparameters, device, show_plot=True, save_path=None):
    num_points_in_dataset = 15
    num_training_points = num_points_in_dataset - 5
    num_datasets = 9

    train_X, train_Y, y_target, _ = prior.get_datasets_from_prior(
        num_datasets, num_points_in_dataset, 1, **prior_hyperparameters
    )
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    y_target = y_target.to(device)
    
    show_vanilla_pfn_predictions(model, train_X, train_Y, num_datasets, num_training_points, device, show_plot=show_plot, save_path=save_path)

    num_training_points = num_points_in_dataset - 5
    num_datasets = 9

    train_X, train_Y, y_target, _ = prior.get_datasets_from_prior(
        num_datasets, num_points_in_dataset, 1, **prior_hyperparameters
    )
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    y_target = y_target.to(device)

def show_prior_pfn_predictions(model, outputs, threshold, parameters, gt_parameters, show=True, save_path=None):
    num_distributions = len(parameters)
    borders_all = model.criterion.borders.detach().cpu().numpy()
    widths = np.diff(borders_all)
    plt.close()
    for i in range(num_distributions):

        left_edges = borders_all[:-1]
        values = torch.squeeze(outputs[i]).detach().cpu().numpy()
        mask = values > threshold
        
        
        fig = plt.figure()
        plt.bar(left_edges[mask], values[mask], width=widths[mask], align='edge', edgecolor='black', linewidth=0.5)

        plt.xlabel(parameters[i])
        plt.ylabel("Probability")
        plt.title(f"Predicted Prob. on GT={gt_parameters[i]}")
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        if show is True: 
            plt.show()

def save_prior_pfn_predictions(model, prior, prior_hyperparameters, device, show_plot=True, save_path=None):
    num_points_in_dataset = 25

    train_X, train_Y, y_target, prior_parameters = prior.get_datasets_from_prior(
        1, num_points_in_dataset, 1, **prior_hyperparameters
    )
    
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    y_target = y_target.to(device)
    
    outputs = evaluate_parameter_distributions_on_model(
        train_X, train_Y, model, device, num_points_in_dataset
    )
    
    show_prior_pfn_predictions(model, outputs, 0.01, ["Prior Parameters"] * len(prior_parameters), prior_parameters, show=show_plot, save_path=save_path)
