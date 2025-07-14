"""
Training Utilities
------------------
This module provides utility functions for advanced training, such as
gradient logging and visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

sns.set_style("whitegrid")

def log_gradients(model: torch.nn.Module, epoch: int, gradient_history: dict):
    """
    Logs the L2 norm of gradients for each named parameter in the model.

    Args:
        model (torch.nn.Module): The model being trained.
        epoch (int): The current epoch number.
        gradient_history (dict): A dictionary to store the gradient norms.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if name not in gradient_history:
                gradient_history[name] = []
            gradient_history[name].append((epoch, grad_norm))

def plot_gradient_flow(gradient_history: dict, output_path: str):
    """
    Plots the gradient flow for a selection of layers.

    Args:
        gradient_history (dict): History of gradient norms for each layer.
        output_path (str): Path to save the plot image.
    """
    if not gradient_history:
        print("Gradient history is empty. Skipping plot.")
        return

    plt.figure(figsize=(15, 10))
    
    # Heuristic to select a manageable number of layers to plot
    plot_every_n = max(1, len(gradient_history) // 20)
    
    layers_to_plot = list(gradient_history.keys())[::plot_every_n]
    
    for i, name in enumerate(layers_to_plot):
        epochs, norms = zip(*gradient_history[name])
        plt.plot(epochs, norms, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Flow Over Epochs")
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Gradient flow plot saved to: {output_path}") 