import pandas as pd
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_root", type=str, default="./tcr_mlm_lora_output/")
args = parser.parse_args()

checkpoint_root = args.checkpoint_root

def cal_three_index(csv_path, full_path):
    print(f"Processing CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df[["umap1", "umap2"]].values

    cluster_keys = ["V", "J", "Epitope species", "CDR3_length"]

    results = []
    for key in cluster_keys:
        labels = df[key]
        
        if labels.dtype.kind in {"O", "U", "S"}:
            labels_enc = LabelEncoder().fit_transform(labels)
        else:
            labels_enc = LabelEncoder().fit_transform(labels.astype(str))
        
        # DB
        db_score = davies_bouldin_score(X, labels_enc)
        
        # intra-class mean distance,variance
        unique_labels = np.unique(labels_enc)
        intra_class_distances = []
        intra_class_variances = []

        for lbl in unique_labels:
            class_points = X[labels_enc == lbl]
            n = len(class_points)
            if n <= 1:
                continue
            
            pairwise_dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(class_points[i] - class_points[j])
                    pairwise_dists.append(dist)
            mean_dist = np.mean(pairwise_dists)
            intra_class_distances.append(mean_dist)
            
            if n > 1:
                cov = np.cov(class_points.T)
                variance = np.trace(cov)
                intra_class_variances.append(variance)
        
        avg_intra_class_distance = np.mean(intra_class_distances)
        avg_intra_class_variance = np.mean(intra_class_variances)

        results.append({
            "Cluster_Key": key,
            "DB_Index": db_score,
            "IntraClass_MeanDistance": avg_intra_class_distance,
            "IntraClass_Variance": avg_intra_class_variance
        })

    results_df = pd.DataFrame(results)
    save_path = os.path.join(full_path, "cluster_metrics.csv")
    results_df.to_csv(save_path, index=False)

    print("Clustering metrics have been saved to cluster_metrics.csv file...")
def main():
    for subdir in os.listdir(checkpoint_root):
        full_path = os.path.join(checkpoint_root, subdir)
        if os.path.isdir(full_path):
            csv_path = os.path.join(full_path, "embedding_umap.csv")
            if os.path.exists(csv_path):
                cal_three_index(csv_path, full_path)

if __name__ == "__main__":
    main()

