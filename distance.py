import os
import glob
import argparse
import numpy as np

"""
Place your .npy files into a folder (e.g., ./data) following the naming rules:
category_0.npy, category_1.npy, ...  A NumPy array of shape (N, D), representing all sample vectors of class m.
sample_0_0.npy, sample_0_1.npy, sample_1_0.npy ....  A single sample vector (shape (D,)) belonging to class i, and is the j-th sample.
"""

def compute_category_centers(data_dir):
    """
    Load all category_*.npy files and compute the mean (center) for each class.
    Returns a dictionary: class_id (str) -> center vector (np.ndarray).
    """
    centers = {}
    category_paths = glob.glob(os.path.join(data_dir, "category_*.npy"))
    for path in category_paths:
        filename = os.path.basename(path)
        class_id = filename.split("_")[1].split(".")[0]
        vectors = np.load(path)
        centers[class_id] = vectors.mean(axis=0)
    return centers

def compute_distances_to_centers(data_dir, centers):
    """
    Compute Euclidean distances from each sample_i_j.npy to its corresponding class center.
    Returns a list of strings formatted as: "filename<TAB>distance"
    """
    results = []
    sample_paths = glob.glob(os.path.join(data_dir, "sample_*_*.npy"))
    for path in sample_paths:
        filename = os.path.basename(path)
        parts = filename.split("_")  # e.g., sample_2_10.npy
        if len(parts) != 3 or not parts[2].endswith(".npy"):
            continue  # skip invalid files
        class_id = parts[1]
        vector = np.load(path)
        center = centers.get(class_id)
        if center is None:
            raise ValueError(f"No category center found for class ID '{class_id}'")
        distance = np.linalg.norm(vector - center)
        results.append(f"{filename}\t{distance:.6f}\n")
    return results

def main(params):
    # Compute category centers
    centers = compute_category_centers(params.data_dir)
    print(f"Loaded {len(centers)} category centers.")

    # Compute distances
    results = compute_distances_to_centers(params.data_dir, centers)

    # Optionally sort results by distance
    if params.sort:
        results.sort(key=lambda s: float(s.split("\t")[1]))

    # Write to output file
    with open(params.output_file, "w", encoding="utf-8") as f:
        f.writelines(results)

    print(f"Completed. {len(results)} distances written to {params.output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute distances from samples to category centers.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .npy files.")
    parser.add_argument("--output_file", type=str, default="dist.out", help="Output file to save distances.")
    parser.add_argument("--sort", action="store_true", help="Sort samples by distance before saving.")
    params = parser.parse_args()
    main(params)
