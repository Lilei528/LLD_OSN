import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


def load_and_process_probs(npy_file, num_topk):
    """
    Load prediction probability matrix and extract Top-K sorted probabilities per method.

    Args:
        npy_file (str): Path to the .npy file.
        num_topk (int): Number of top probabilities to extract.

    Returns:
        topk_probs (np.ndarray): Array of shape (num_methods, num_topk).
    """
    probs = np.load(npy_file)  # shape: (num_methods, num_classes)
    if probs.ndim != 2:
        raise ValueError("Input array must be 2D: (num_methods, num_classes)")

    if probs.shape[1] < num_topk:
        raise ValueError(f"Number of classes ({probs.shape[1]}) is less than num_topk ({num_topk})")

    # Extract Top-K class probabilities for each method
    topk = np.sort(probs, axis=1)[:, -num_topk:]  # sort and take last k
    return topk[:, ::-1]  # reverse to descending order


def plot_heatmap(topk_probs, baseline_names, class_names, title, save_path=None):
    """
    Plot a heatmap of Top-K class probabilities for each baseline method.

    Args:
        topk_probs (np.ndarray): Array of shape (num_methods, num_topk).
        baseline_names (list): List of baseline method names.
        class_names (list): List of top-k class labels.
        title (str): Title for the heatmap.
        save_path (str): Optional path to save the figure.
    """
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(10, 6))

    sns.heatmap(topk_probs, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=class_names, yticklabels=baseline_names, vmin=0, vmax=1)

    plt.title(title)
    plt.xlabel("Top-K Classes")
    plt.ylabel("Baselines")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()


def main(params):
    baseline_names = params.baseline_names.split(",")
    class_names = [f"Top-{i + 1}" for i in range(params.topk)]

    topk_probs = load_and_process_probs(params.npy_file, params.topk)

    plot_heatmap(
        topk_probs=topk_probs,
        baseline_names=baseline_names,
        class_names=class_names,
        title=params.title,
        save_path=params.save_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Top-K Class Probability Heatmap for Multiple Baselines")
    parser.add_argument("--npy_file", type=str, required=True,
                        help="Path to input .npy file of shape (methods, classes)")
    parser.add_argument("--topk", type=int, default=5, help="Number of top classes to display per method")
    parser.add_argument("--baseline_names", type=str, default="Selfmix,Co - teaching,PNP,Noise Matrix,Towrad,LLD - OSN",
                        help="Comma-separated baseline names")
    parser.add_argument("--title", type=str, default="Top-K Class Probabilities for Different Baselines",
                        help="Plot title")
    parser.add_argument("--save_path", type=str, help="Optional: file path to save the plot (e.g., heatmap.png)")
    args = parser.parse_args()
    main(args)
