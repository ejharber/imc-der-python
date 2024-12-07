import os
import torch
import matplotlib.pyplot as plt

def plot_losses(
    model_paths,
    model_names,
    title="Loss Curves (Log Scale)",
    font_size=12,
    title_font_size=14,
    legend_font_size=10
):
    """
    Load training, validation, and test losses from multiple model checkpoints and plot them
    with a log scale on the y-axis.

    :param model_paths: List of paths to the model checkpoint files.
    :param model_names: List of names corresponding to the models for the legend.
    :param title: Custom title for the plot.
    :param font_size: Font size for axis labels.
    :param title_font_size: Font size for the title.
    :param legend_font_size: Font size for the legend.
    """
    if len(model_paths) != len(model_names):
        print("Error: The number of model paths and model names must match.")
        return

    plt.figure(figsize=(12, 8), dpi=100)
    
    # Define colors for models
    colors = ['blue', 'orange']  # First color for Position, second for Position + Force
    
    for idx, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        if not os.path.exists(model_path):
            print(f"File not found: {model_path}")
            continue
        
        # Load the checkpoint, mapping to CPU if necessary
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            continue
        
        # Ensure necessary keys exist in the checkpoint
        for key in ['train_losses', 'valid_losses', 'test_losses']:
            if key not in checkpoint:
                print(f"Warning: No '{key}' found in {model_path}. Skipping this loss.")
                continue

        # Set the color for the current model
        color = colors[idx % len(colors)]

        # Plot training losses with a solid line
        if 'train_losses' in checkpoint:
            plt.plot(
                checkpoint['train_losses'],
                label=f"{model_name}: Training",
                linestyle='-', color=color
            )
        
        # Plot test losses with a dashed line
        if 'test_losses' in checkpoint:
            plt.plot(
                checkpoint['test_losses'],
                label=f"{model_name}: Test",
                linestyle='--', color=color
            )
    
        # Plot validation losses with a solid line and circle markers
        if 'valid_losses' in checkpoint:
            print(checkpoint['valid_losses'][-1])
            plt.plot(
                checkpoint['valid_losses'],
                label=f"{model_name}: Validation",
                linestyle='-', marker='o', color=color
            )

    # Customize plot with font sizes
    plt.title(title, fontsize=title_font_size)
    plt.xlabel("Num Epochs", fontsize=font_size)
    plt.ylabel("Log MSE Loss", fontsize=font_size)
    plt.yscale('log')  # Set y-axis to log scale
    plt.legend(fontsize=legend_font_size)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

# Example usage for zeroshot data
model_paths = [
    '../../4_SupervizedLearning/zero_shot/checkpoints_N2_pose/final_model_checkpoint.pth',
    '../../4_SupervizedLearning/zero_shot/checkpoints_N2_all/final_model_checkpoint.pth',
]

model_names = [
    'Position',
    'Position + Force',
]

plot_losses(
    model_paths, 
    model_names,
    title="Zeroshot Policy Loss Curves",
    font_size=14,
    title_font_size=16,
    legend_font_size=12
)

# Example usage for iterative data
model_paths = [
    '../../4_SupervizedLearning/iterative/checkpoints_dgoal_daction_noise_N2_pose_large_new/model_checkpoint_200.pth',
    '../../4_SupervizedLearning/iterative/checkpoints_dgoal_daction_noise_N2_all_large_new/model_checkpoint_200.pth',
]

model_names = [
    'Position',
    'Position + Force',
]

plot_losses(
    model_paths, 
    model_names,
    title="Iterative Policy Loss Curves",
    font_size=14,
    title_font_size=16,
    legend_font_size=12
)
