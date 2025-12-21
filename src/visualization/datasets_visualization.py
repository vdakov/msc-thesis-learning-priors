import matplotlib.pyplot as plt 
import seaborn as sns 

def show_datasets(x, y, y_noisy, title):
    _, axs = plt.subplots(1, 2, figsize=(15, 5))
    x = x.squeeze().T
    y = y.squeeze().T
    y_noisy = y_noisy.squeeze().T
    for i in range(len(x)):
        axs[0].scatter(x[i], y[i])
        axs[1].scatter(x[i], y_noisy[i])
    axs[0].set_title("Clean Datasets")
    
    axs[1].set_title("Noisy Datasets")
    plt.title(title)
    plt.show()
    