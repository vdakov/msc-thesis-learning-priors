import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

def plot_training_validation_loss(losses, val_losses, title="Training and Validation Loss"):
    
    plt.figure(figsize=(15, 5))
    sns.lineplot(x=np.arange(0, len(losses)), y=np.array(losses), label="Training")
    if val_losses:
        sns.lineplot(x=np.arange(0, len(losses)), y=np.array(val_losses), label="Validation")
    else: 
        title ='Training Loss'
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.ylabel('Criterion')
    plt.xlabel('Epochs')
    plt.show()
    
def save_training_validation_loss(losses, val_losses, save_path, title="Training and Validation Loss"):
    plt.figure(figsize=(15, 5))
    sns.lineplot(x=np.arange(0, len(losses)), y=np.array(losses), label="Training")
    if val_losses:
        sns.lineplot(x=np.arange(0, len(losses)), y=np.array(val_losses), label="Validation")
    else: 
        title ='Training Loss'
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.ylabel('Criterion')
    plt.xlabel('Epochs')
    plt.savefig(save_path)
    plt.close()