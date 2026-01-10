import torch 
import matplotlib.pyplot as plt 

def show_vanilla_pfn_predictions(model, train_X, train_Y, num_datasets, num_training_points, device):

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
            test_x[order_test_x, 0].cpu().numpy(),
            pred_confs[order_test_x][:, 0].cpu().numpy(),
            pred_confs[order_test_x][:, 1].cpu().numpy(),
            alpha=.1,
            color='green'
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.legend()
    plt.show()


    