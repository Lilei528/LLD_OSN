import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_method_names(data_dir):
    """
    Extract method names from files in the given directory.
    Looks for pairs of 'pre_<method>.npy' and 'preood_<method>.npy'.
    """
    pre_files = [f for f in os.listdir(data_dir)
                 if f.startswith("pre_") and not f.startswith("preood_") and f.endswith(".npy")]

    method_names = []
    for file in pre_files:
        method = file[len("pre_"):-len(".npy")]
        ood_file = f"preood_{method}.npy"
        if os.path.exists(os.path.join(data_dir, ood_file)):
            method_names.append(method)
        else:
            print(f"[WARN] OOD file missing for method: {method}")

    return sorted(method_names)


def compute_confidences(method_names, data_dir):
    """
    Compute the average confidence (max probability) for in-distribution and out-of-distribution samples.
    Returns a dictionary of method names and corresponding confidence values.
    """
    results = {'Method': [], 'Test Samples': [], 'OOD Samples': []}

    for method in method_names:
        in_path = os.path.join(data_dir, f"pre_{method}.npy")
        ood_path = os.path.join(data_dir, f"preood_{method}.npy")

        try:
            in_probs = np.load(in_path)
            ood_probs = np.load(ood_path)

            in_conf = float(np.mean(np.max(in_probs, axis=1)))
            ood_conf = float(np.mean(np.max(ood_probs, axis=1)))

            results['Method'].append(method)
            results['Test Samples'].append(in_conf)
            results['OOD Samples'].append(ood_conf)
        except Exception as e:
            print(f"[ERROR] Failed to process method '{method}': {e}")

    return results


def plot_confidence_bar(data_dict, save_path):
    """
    Generate and save a grouped bar plot comparing confidence on in-distribution and out-of-distribution samples.
    """
    df = pd.DataFrame(data_dict)
    melted_df = pd.melt(df, id_vars=['Method'],
                        value_vars=['Test Samples', 'OOD Samples'],
                        var_name='Type', value_name='Confidence')

    bar_colors = ['#BFBCD8', '#F7B2AC']

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Method', y='Confidence', hue='Type', data=melted_df, palette=bar_colors)

    plt.legend(bbox_to_anchor=(0.025, 0.75), loc='lower left', prop={'size': 20})
    plt.xticks(rotation=0, ha='center', fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('', fontsize=20)
    plt.ylabel('Confidence', fontsize=22)
    plt.legend(fontsize=22)
    ax.tick_params(axis='y', labelsize=22)

    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.show()


def main(params):
    """
    Main function to compute average confidence and generate bar plot.
    Expects:
    - input_dir: directory containing pre_<method>.npy and preood_<method>.npy
    - output_path: path to save the output figure
    """
    input_dir = params.input_dir
    output_path = params.output_path

    method_list = extract_method_names(input_dir)
    if not method_list:
        print("[ERROR] No valid method files found.")
        return

    confidence_data = compute_confidences(method_list, input_dir)
    plot_confidence_bar(confidence_data, output_path)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .npy files.")
    parser.add_argument("--output_path", type=str, default="dist.out", help="Output file to save distances.")
    params = parser.parse_args()
    main(params)
