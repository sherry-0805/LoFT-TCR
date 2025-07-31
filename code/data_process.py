import pandas as pd
import numpy as np
import os, sys
import pickle
import torch
import torchmetrics
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from config import *

import dgl
import umap
from sklearn.preprocessing import StandardScaler
root = os.path.join(args.pridir, args.secdir, args.terdir)
train_csv = os.path.join(root, 'train.tsv')
test_csv = os.path.join(root, 'test.tsv')
train_delimiter = '\t'
test_delimiter = '\t'

path_pickle_train = os.path.join(root, 'global_train_dataset_esm2feature.pickle')
path_pickle_test = os.path.join(root, 'global_test_dataset_esm2feature.pickle')
train_path_pickle_cdr3 = os.path.join(root, 'esm2_feature_cdr3_train.pickle')
train_path_pickle_peptide = os.path.join(root, 'esm2_feature_peptide_train.pickle')
test_path_pickle_cdr3 = os.path.join(root, 'esm2_feature_cdr3_test.pickle')
test_path_pickle_peptide = os.path.join(root, 'esm2_feature_peptide_test.pickle')


def TCRDataset_global(cdr3b_map, peptide_map, cdr3b_graph, peptide_graph, edge_index_pos):
        maxlength = 200
        Graphdata = HeteroData()
        cdr3b_x = torch.Tensor()
        peptide_x = torch.Tensor()
        cdr3b_feature_map = {}
        for cb, cb_num in cdr3b_map.items():
            cdr3b_feature = cdr3b_graph[cb]
            if isinstance(cdr3b_feature, torch.Tensor):
                cdr3b_feature = cdr3b_feature.cpu().numpy()
            cdr3b_feature_map[cb_num] = cdr3b_feature
        for i in range(len(cdr3b_feature_map)):
            cdr3b_x = torch.cat((cdr3b_x, torch.Tensor(cdr3b_feature_map[i]).unsqueeze(0)), 0)
            
        peptide_feature_map = {}
        for pep, pep_num in peptide_map.items():
            peptide_feature = peptide_graph[pep]
            if isinstance(peptide_feature, torch.Tensor):
                peptide_feature= peptide_feature.cpu().numpy()
            peptide_feature_map[pep_num] = peptide_feature
        for i in range(len(peptide_feature_map)):
            peptide_x = torch.cat((peptide_x, torch.Tensor(peptide_feature_map[i]).unsqueeze(0)), 0)

        Graphdata['cdr3b'].x = cdr3b_x
        Graphdata['peptide'].x = peptide_x
        Graphdata['cdr3b', 'CBindA', 'peptide'].edge_index = edge_index_pos
        Graphdata = ToUndirected()(Graphdata)
        return Graphdata


def create_dataset_global():
    if not os.path.exists(root):
        os.makedirs(root)
    else:
        print(root)

    # train data
    train_data = pd.read_csv(train_csv, delimiter=train_delimiter)
    train_cdr3b, train_peptide, train_binder = list(train_data['cdr3']), list(train_data['peptide']), list(train_data['Binding'])

    train_cdr3b_unique_list = list(train_data['cdr3'].unique())
    train_peptide_unique_list = list(train_data['peptide'].unique())

    mapping_train_cdr3 = {cdr3_name: i for i, cdr3_name in enumerate(train_cdr3b_unique_list)}
    mapping_train_peptide = {peptide_name: i for i, peptide_name in enumerate(train_peptide_unique_list)}
    train_src = [mapping_train_cdr3[train_cdr3b[cci]] for cci in range(len(train_cdr3b))]
    train_dst = [mapping_train_peptide[train_peptide[ppi]] for ppi in range(len(train_peptide))]
    train_edge_index = torch.tensor([train_src, train_dst])


    print("loading global train dataset...")
    if not os.path.exists(path_pickle_train):

        required_files = [train_path_pickle_cdr3, train_path_pickle_peptide]
        for fpath in required_files:
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Required feature file {fpath} missing!")

        with open(train_path_pickle_cdr3, 'rb') as f1:
            cdr3b_graph = pickle.load(f1)
        with open(train_path_pickle_peptide, 'rb') as f2:
            peptide_graph = pickle.load(f2)

        train_dataset = TCRDataset_global(
            cdr3b_map=mapping_train_cdr3,
            peptide_map=mapping_train_peptide,
            cdr3b_graph=cdr3b_graph,
            peptide_graph=peptide_graph,
            edge_index_pos=train_edge_index
        )

        try:
            with open(path_pickle_train, 'wb') as f_out:
                pickle.dump(train_dataset, f_out)
            print(f"Successfully created and saved training dataset to {path_pickle_train}")
        except Exception as e:
            print(f"Save failed: {str(e)}")
            raise
    else:
        with open(path_pickle_train, 'rb') as f1:
            train_dataset = pickle.load(f1)
        print("train dataset global pickle has loaded")
    print("train dataset global has prepared")
    torch.cuda.empty_cache()
    # test data
    test_data = pd.read_csv(test_csv, delimiter=test_delimiter)
    test_cdr3b, test_peptide, test_binder = list(test_data['cdr3']), list(test_data['peptide']), list(test_data['Binding'])

    test_cdr3b_unique_list = list(test_data['cdr3'].unique())
    test_peptide_unique_list = list(test_data['peptide'].unique())

    mapping_test_cdr3 = {cdr3_name: i for i, cdr3_name in enumerate(test_cdr3b_unique_list)}
    # num_cdr3b_nodes_test = len(test_cdr3b_unique_list)
    # mapping_test_peptide = {peptide_name: i + num_cdr3b_nodes_test for i, peptide_name in enumerate(test_peptide_unique_list)}
    mapping_test_peptide = {peptide_name: i for i, peptide_name in enumerate(test_peptide_unique_list)}
    test_src = [mapping_test_cdr3[test_cdr3b[cci]] for cci in range(len(test_cdr3b))]
    test_dst = [mapping_test_peptide[test_peptide[ppi]] for ppi in range(len(test_peptide))]
    test_edge_index = torch.tensor([test_src, test_dst])


    print("loading global test dataset...")
    if not os.path.exists(path_pickle_test):
        with open(test_path_pickle_cdr3, 'rb') as f3:
            test_cdr3b_graph = pickle.load(f3)
        with open(test_path_pickle_peptide, 'rb') as f4:
            test_peptide_graph = pickle.load(f4)

        test_cdr3b, test_peptide, test_binder = np.asarray(test_cdr3b), np.asarray(test_peptide), np.asarray(test_binder)

        with open(path_pickle_test, 'wb') as f2:
            test_dataset = TCRDataset_global(cdr3b_map=mapping_test_cdr3, peptide_map=mapping_test_peptide,
                                       cdr3b_graph=test_cdr3b_graph, peptide_graph=test_peptide_graph,
                                       edge_index_pos=test_edge_index)
            pickle.dump(test_dataset, f2)
        with open(path_pickle_test, 'rb') as f2:
            test_dataset = pickle.load(f2)
            print("test dataset pickle saved")
    else:
        with open(path_pickle_test, 'rb') as f2:
            test_dataset = pickle.load(f2)
        print("test dataset global pickle has loaded")
    print("test dataset global has prepared")
    torch.cuda.empty_cache()
    return train_dataset, test_dataset, train_edge_index, test_edge_index, train_binder, test_binder

def perform_umap(features,
                umap_n_neighbors=15, 
                umap_min_dist=0.1, 
                umap_metric='euclidean', 
                random_state=42):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("UMAP...")
    reducer = umap.UMAP(n_components=2, 
                            n_neighbors=umap_n_neighbors, 
                            min_dist=umap_min_dist, 
                            metric=umap_metric, 
                            random_state=random_state)
    umap_features = reducer.fit_transform(scaled_features)
    return umap_features


def umap_and_save_csv(paired_features, cdr3_sequences, pep_sequences, binding_values, output_csv):
    umap_results = perform_umap(paired_features)

    results_df = pd.DataFrame({
        'peptide': pep_sequences,
        'cdr3': cdr3_sequences,
        'Binding': binding_values,
        'UMAP_1': umap_results[:, 0],
        'UMAP_2': umap_results[:, 1]
    })

    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def extract_features_and_sequences(test_dataset, mapping_test_cdr3, mapping_test_peptide, test_data):
    cdr3b_features = test_dataset['cdr3b'].x  #(num_cdr3b_nodes, feature_dim)
    peptide_features = test_dataset['peptide'].x  #(num_peptide_nodes, feature_dim)

    edge_index = test_dataset['cdr3b', 'CBindA', 'peptide'].edge_index  # (2, num_edges)
    
    paired_features = []
    cdr3_sequences = []
    pep_sequences = [] 
    binding_values = []

    reverse_mapping_cdr3 = {v: k for k, v in mapping_test_cdr3.items()}
    reverse_mapping_peptide = {v: k for k, v in mapping_test_peptide.items()}

    binding_map = {}
    for _, row in test_data.iterrows():
        cdr3_seq = row['cdr3']
        pep_seq = row['peptide']
        binding_value = row['Binding']
        binding_map[f"{cdr3_seq}_{pep_seq}"] = binding_value

    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        cdr3_idx = edge_index[0, i].item()
        pep_idx = edge_index[1, i].item()

        cdr3_seq = reverse_mapping_cdr3[cdr3_idx]
        pep_seq = reverse_mapping_peptide[pep_idx]

        cdr3_feature = cdr3b_features[cdr3_idx] 
        pep_feature = peptide_features[pep_idx]
        paired_feature = torch.cat([cdr3_feature, pep_feature], dim=0)
        paired_features.append(paired_feature.numpy())

        binding_key = f"{cdr3_seq}_{pep_seq}"
        if binding_key in binding_map:
            binding_value = binding_map[binding_key]
        else:
            binding_value = np.nan

        cdr3_sequences.append(cdr3_seq)
        pep_sequences.append(pep_seq)
        binding_values.append(binding_value)

    paired_features = np.array(paired_features)
    cdr3_sequences = np.array(cdr3_sequences)
    pep_sequences = np.array(pep_sequences)
    binding_values = np.array(binding_values)

    return paired_features, cdr3_sequences, pep_sequences, binding_values

def create_dataset_global_predict():
    if not os.path.exists(root):
        os.makedirs(root)
        
    # test data
    test_data = pd.read_csv(test_csv, delimiter=test_delimiter)
    test_cdr3b, test_peptide, test_binder = list(test_data['cdr3']), list(test_data['peptide']), list(test_data['Binding'])

    test_cdr3b_unique_list = list(test_data['cdr3'].unique())
    test_peptide_unique_list = list(test_data['peptide'].unique())

    mapping_test_cdr3 = {cdr3_name: i for i, cdr3_name in enumerate(test_cdr3b_unique_list)}
    mapping_test_peptide = {peptide_name: i for i, peptide_name in enumerate(test_peptide_unique_list)}
    
    test_src = [mapping_test_cdr3[test_cdr3b[cci]] for cci in range(len(test_cdr3b))]
    test_dst = [mapping_test_peptide[test_peptide[ppi]] for ppi in range(len(test_peptide))]
    test_edge_index = torch.tensor([test_src, test_dst])


    print("loading global test dataset...")
    if not os.path.exists(path_pickle_test):
        with open(test_path_pickle_cdr3, 'rb') as f3:
            test_cdr3b_graph = pickle.load(f3)
        with open(test_path_pickle_peptide, 'rb') as f4:
            test_peptide_graph = pickle.load(f4)

        test_cdr3b, test_peptide, test_binder = np.asarray(test_cdr3b), np.asarray(test_peptide), np.asarray(test_binder)

        with open(path_pickle_test, 'wb') as f2:
            test_dataset = TCRDataset_global(cdr3b_map=mapping_test_cdr3, peptide_map=mapping_test_peptide,
                                       cdr3b_graph=test_cdr3b_graph, peptide_graph=test_peptide_graph,
                                       edge_index_pos=test_edge_index)
            pickle.dump(test_dataset, f2)
        with open(path_pickle_test, 'rb') as f2:
            test_dataset = pickle.load(f2)
            print("test dataset pickle saved")
    else:
        with open(path_pickle_test, 'rb') as f2:
            test_dataset = pickle.load(f2)
        print("test dataset global pickle has loaded")
    print("test dataset global has prepared")
    # Peptide-CDR features encoded by ESM2
    '''
    output_csv = os.path.join(root, "ESM2_umap.csv")
    paired_features, cdr3_sequences, pep_sequences, binding_values = extract_features_and_sequences(
        test_dataset, mapping_test_cdr3, mapping_test_peptide, test_data
    )
    umap_and_save_csv(paired_features, cdr3_sequences, pep_sequences, binding_values, output_csv)
    '''
    return test_dataset, test_edge_index, test_binder