import os
import torch
import matplotlib.pyplot as plt

def plot_validation_losses(model_paths, model_names):
    """
    Load validation losses from multiple model checkpoints and plot them with a log scale on the y-axis.

    :param model_paths: List of paths to the model checkpoint files.
    :param model_names: List of names corresponding to the models for the legend.
    """
    if len(model_paths) != len(model_names):
        print("Error: The number of model paths and model names must match.")
        return
    
    plt.figure(figsize=(10, 6))
    
    for idx, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        if not os.path.exists(model_path):
            print(f"File not found: {model_path}")
            continue
        
        # Load the checkpoint
        checkpoint = torch.load(model_path)
        
        # Ensure 'valid_losses' exists in the checkpoint
        if 'valid_losses' not in checkpoint:
            print(f"No 'valid_losses' found in {model_path}")
            continue
        
        valid_losses = checkpoint['valid_losses']
        
        # Plot the validation losses with a label
        plt.plot(valid_losses, label=model_name)
    
    plt.title("Validation Loss Curves (Log Scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (Log Scale)")
    plt.yscale('log')  # Set y-axis to log scale
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.tight_layout()
    plt.show()

# Example usage
model_paths = [
    '../4_SupervizedLearning/iterative/checkpoints_N2_all/final_model_checkpoint.pth',
    '../4_SupervizedLearning/iterative/checkpoints_N2_pose/final_model_checkpoint.pth',
    '../4_SupervizedLearning/iterative/checkpoints_nonoise_N2_all/final_model_checkpoint.pth',
    '../4_SupervizedLearning/iterative/checkpoints_nonoise_N2_pose/final_model_checkpoint.pth',
]

model_names = [
    'N2 All',
    'N2 Pose',
    'N2 All (No Noise)',
    'N2 Pose (No Noise)'
]

# # Example usage
# model_paths = [
#     '../4_SupervizedLearning/zero_shot/checkpoints_N2_all/final_model_checkpoint.pth',
#     '../4_SupervizedLearning/zero_shot/checkpoints_N2_pose/final_model_checkpoint.pth',
# ]

# model_names = [
#     'N2 All',
#     'N2 Pose',
# ]

plot_validation_losses(model_paths, model_names)
