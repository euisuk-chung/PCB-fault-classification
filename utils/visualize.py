import matplotlib.pyplot as plt
import numpy as np

def plot_result(model_index, fold, train_loss, train_acc, val_loss, val_acc):
    epochs = list(range(1, len(train_loss)+1))
    
    # Draw loss plot
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xticks(range(1, len(train_loss)+1))
    plt.legend(loc='best')
    plt.savefig(f'./plots/loss_model{model_index}_fold{fold}.png')
    plt.clf()
    
    # Draw acc plot
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xticks(range(1, len(train_loss)+1))
    plt.legend(loc='best')
    plt.savefig(f'./plots/acc_model{model_index}_fold{fold}.png')
    
    return
