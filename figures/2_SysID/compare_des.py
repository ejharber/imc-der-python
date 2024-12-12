import numpy as np
import matplotlib.pyplot as plt

def plot_cost_comparison(file_path_1, file_path_2):
    """
    Opens two .npz files saved by the SaveCostHistory class, compares their total costs, 
    and calculates the final errors and percentage improvement.

    Args:
        file_path_1 (str): Path to the first .npz file.
        file_path_2 (str): Path to the second .npz file.
    """
    # Load the data from both files
    data1 = np.load(file_path_1)
    data2 = np.load(file_path_2)

    # Extract total costs
    total_costs_1 = data1.get("total_costs", None)
    total_costs_2 = data2.get("total_costs", None)

    if total_costs_1 is None or total_costs_2 is None:
        print("Error: One or both files do not contain 'total_costs'.")
        return

    # Ensure the lengths match for comparison
    min_length = min(len(total_costs_1), len(total_costs_2))
    total_costs_1 = total_costs_1[:min_length]
    total_costs_2 = total_costs_2[:min_length]

    # Plot the two total costs against each other
    plt.figure(figsize=(8, 6))
    plt.plot(total_costs_1, label='Material Damping', marker='o')
    plt.plot(total_costs_2, label='Original Model', marker='x')
    plt.title("Comparison of Total Costs")
    plt.xlabel("Iteration")
    plt.ylabel("Total Cost")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print final errors
    final_cost_1 = total_costs_1[-1]
    final_cost_2 = total_costs_2[-1]
    print(f"Final Total Cost (Material Damping): {final_cost_1:.4f}")
    print(f"Final Total Cost (Original Model): {final_cost_2:.4f}")

    # Calculate and print percentage improvement
    improvement = ((final_cost_2 - final_cost_1) / final_cost_2) * 100
    print(f"Percentage Improvement: {improvement:.2f}%")

# Example usage
if __name__ == "__main__":
    file1 = "../../2_SysID/params/N2_all_naxial_80.npz"  # Replace with actual path
    file2 = "../../2_SysID/params/N2_all_80.npz"  # Replace with actual path
    plot_cost_comparison(file1, file2)
