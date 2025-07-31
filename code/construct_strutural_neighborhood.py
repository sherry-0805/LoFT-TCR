import dgl
import torch
import numpy as np
from sklearn.manifold import Isomap
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import umap
from sklearn.decomposition import PCA
from collections import Counter
import networkx as nx
from sklearn.neighbors import KDTree

def get_graph_neighbors_dgl(graph):
    neighbors = {}
    for node in range(graph.num_nodes()):
        neighbors[node] = graph.successors(node).tolist()
    return neighbors

def compute_isomap_embeddings(node_features, n_components=2):
    isomap = Isomap(n_components=n_components)
    embeddings = isomap.fit_transform(node_features)
    print('isomap')
    return embeddings


def compute_umap_embeddings(node_features, n_components=2, n_neighbors=15,
                            min_dist=0.1, metric="euclidean"):
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           min_dist=min_dist, metric=metric, random_state =42)
    embeddings = umap_model.fit_transform(node_features)
    print('umap')
    return embeddings


def compute_pca_embeddings(node_features, n_components=2):
    pca = PCA(n_components=n_components,random_state =42)
    embeddings = pca.fit_transform(node_features)
    print('pca')
    return embeddings
from sklearn.neighbors import KDTree
import numpy as np

def get_latent_tcr_neighbors(embeddings, threshold_tcr, num_tcr_nodes, consNeighborType='rho'):
    tcr_neighbors = {}
    tree = KDTree(embeddings[:num_tcr_nodes]) 

    for i in range(num_tcr_nodes):
        if consNeighborType == 'rho':
            indices = tree.query_radius([embeddings[i]], r=threshold_tcr)[0]
        else:
            indices = tree.query([embeddings[i]], k=threshold_tcr + 1) 
        tcr_neighbors[i] = [idx for idx in indices if idx != i]

    return tcr_neighbors

def get_latent_peptide_neighbors(embeddings, threshold_pep, num_tcr_nodes, consNeighborType='rho'):
    peptide_neighbors = {}
    tree = KDTree(embeddings[num_tcr_nodes:])

    for i in range(num_tcr_nodes, len(embeddings)):
        if consNeighborType == 'rho':
            indices = tree.query_radius([embeddings[i]], r=threshold_pep)[0]
        else:
            indices = tree.query([embeddings[i]], k=threshold_pep + 1)
        peptide_neighbors[i] = [idx + num_tcr_nodes for idx in indices if idx + num_tcr_nodes != i]

    return peptide_neighbors

def find_threshold_by_type(embeddings, graph_neighbors, num_tcr_nodes, consNeighborType='rho'):
    avg_graph_neighbors_tcr = sum(
        len(graph_neighbors[node]) for node in range(num_tcr_nodes)
    ) / num_tcr_nodes
    avg_graph_neighbors_peptide = sum(
        len(graph_neighbors[node]) for node in range(num_tcr_nodes, len(embeddings))
    ) / (len(embeddings) - num_tcr_nodes)

    print(f"Average graph neighbors (TCR): {avg_graph_neighbors_tcr}")
    print(f"Average graph neighbors (Peptide): {avg_graph_neighbors_peptide}")

    if consNeighborType == 'rho':
        threshold_tcr = 0.01
        threshold_pep = 0.1
    else :
        threshold_tcr = 1
        threshold_pep = 30

    while True:
        latent_neighbors_tcr = get_latent_tcr_neighbors(embeddings, threshold_tcr, num_tcr_nodes, consNeighborType = consNeighborType)
        avg_latent_neighbors_tcr = np.mean([len(neigh) for neigh in latent_neighbors_tcr.values()])
        if avg_latent_neighbors_tcr >= avg_graph_neighbors_tcr:
            break
        if consNeighborType == 'rho':
            threshold_tcr += 0.005
        else:
            threshold_tcr += 1
        print(f"tcr threshold：{threshold_tcr}, avg_latent_neighbors_tcr：{avg_latent_neighbors_tcr}, avg_graph_neighbors_tcr：{avg_graph_neighbors_tcr}")

    while True:
        latent_neighbors_peptide = get_latent_peptide_neighbors(embeddings, threshold_pep, num_tcr_nodes, consNeighborType = consNeighborType)
        avg_latent_neighbors_peptide = np.mean([len(neigh) for neigh in latent_neighbors_peptide.values()])
        if avg_latent_neighbors_peptide >= avg_graph_neighbors_peptide:
            break
        if consNeighborType == 'rho':
            threshold_pep += 0.005
        else:
            threshold_pep += 1
        print(f"pep threshold：{threshold_pep}, avg_latent_neighbors_peptide：{avg_latent_neighbors_peptide}, avg_graph_neighbors_peptide：{avg_graph_neighbors_peptide}")

    return threshold_tcr, threshold_pep

def compute_structural_neighborhood(graph, node_features, num_tcr_nodes, dim_reduction='umap', n_components=2, consNeighborType='rho'):
    graph_neighbors = get_graph_neighbors_dgl(graph)

    if dim_reduction == 'umap':
        embeddings = compute_umap_embeddings(node_features.numpy(), n_components=n_components)
    elif dim_reduction == 'isomap':
        embeddings = compute_isomap_embeddings(node_features.numpy(), n_components=n_components)
    elif dim_reduction == 'pca':
        embeddings = compute_pca_embeddings(node_features.numpy(), n_components=n_components)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {dim_reduction}")

    threshold_tcr, threshold_pep = find_threshold_by_type(embeddings, graph_neighbors, num_tcr_nodes, consNeighborType=consNeighborType)
    
    latent_neighbors_tcr = get_latent_tcr_neighbors(embeddings, threshold_tcr, num_tcr_nodes, consNeighborType=consNeighborType)
    latent_neighbors_peptide = get_latent_peptide_neighbors(embeddings, threshold_pep, num_tcr_nodes, consNeighborType=consNeighborType)

    latent_neighbors = {**latent_neighbors_tcr, **latent_neighbors_peptide}

    return graph_neighbors, latent_neighbors, threshold_tcr, threshold_pep
    
def construct_and_concat_edges_with_labels(latent_neighbors):
    latent_src = []
    latent_dst = []
    for node, neighbors in latent_neighbors.items():
        for neighbor in neighbors:
            if node != neighbor:
                latent_src.append(node)
                latent_dst.append(neighbor)

    latent_src = torch.tensor(latent_src, dtype=torch.long)
    latent_dst = torch.tensor(latent_dst, dtype=torch.long)
    latent_edge = torch.stack([latent_src, latent_dst], dim=0)

    # num_graph_edges = edge_index.size(1)
    num_latent_edges = latent_edge.size(1)
    # graph_edge_labels = torch.zeros(num_graph_edges, dtype=torch.long) 
    latent_edge_labels = torch.zeros(num_latent_edges, dtype=torch.long)

    # combined_edge_index = torch.cat([edge_index, latent_edge], dim=1)
    # space_labels = torch.cat([graph_edge_labels, latent_edge_labels], dim=0)
    space_labels = latent_edge_labels
    combined_edge_index = latent_edge
    return combined_edge_index, space_labels


def compute_relationship_index(zv, zu):
    
    if zv[1] <= zu[1]:  
        if zv[0] > zu[0]: 
            return 0 
        else:
            return 1  
    else:
        if zv[0] > zu[0]:
            return 2 
        else:
            return 3  # lower right


def compute_edge_relationships(combined_edge_index, embeddings):
    src_nodes = combined_edge_index[0]
    dst_nodes = combined_edge_index[1]
    relationships = []
    for src, dst in zip(src_nodes, dst_nodes):
        zv = embeddings[src]  
        zu = embeddings[dst]
        relationships.append(compute_relationship_index(zv, zu))
    return relationships

import dgl
import torch as th

def filter_subgraphs_and_concatenate_edges(all_edge_index,subgraph_idx,edge_labels,min_num_nodes = 10):
    subgraph_dict = {}
    num_edges = all_edge_index.shape[1]  # E
    for i in range(num_edges):
        sg_id = subgraph_idx[i]  
        lbl = edge_labels[i]
        src = all_edge_index[0, i].item()
        dst = all_edge_index[1, i].item()
        if sg_id not in subgraph_dict:
            subgraph_dict[sg_id] = {
                'edges': [],
                'edge_labels': []
            }
        subgraph_dict[sg_id]['edges'].append((src, dst))
        subgraph_dict[sg_id]['edge_labels'].append(lbl)
    
    for sg_id, sub_info in subgraph_dict.items():
        num_edges_in_sub = len(sub_info['edges'])
        print(f"Subgraph {sg_id} contains {num_edges_in_sub} edges.")
    
    filtered_edge_indices = []
    filtered_subgraph_indices = []
    filtered_edge_labels = []
    
    for sg_id, sub_info in subgraph_dict.items():
        edges = sub_info['edges']
        labels = sub_info['edge_labels']

        G = nx.Graph()
        G.add_edges_from(edges)

        connected_components = list(nx.connected_components(G))
        for comp in connected_components:

            if len(comp) < min_num_nodes:
                print(f"Subgraph {sg_id}'s connected component {comp} has fewer than {min_num_nodes} nodes and is ignored.")
                continue

            comp_nodes = set(comp)
            comp_edges = []
            comp_edge_labels = []
            for (src, dst), label in zip(edges, labels):
                if src in comp_nodes and dst in comp_nodes:
                    comp_edges.append((src, dst))
                    comp_edge_labels.append(label)

            if len(comp_edges) == 0:
                print(f"Connected component {comp} in subgraph {sg_id} has no edges and is ignored.")
                continue

            comp_src = [e[0] for e in comp_edges]
            comp_dst = [e[1] for e in comp_edges]
            comp_edge_index = torch.tensor([comp_src, comp_dst], dtype=torch.long)

            filtered_edge_indices.append(comp_edge_index)

            filtered_subgraph_indices.append(torch.full((comp_edge_index.shape[1],), sg_id, dtype=torch.long))
            filtered_edge_labels.append(torch.tensor(comp_edge_labels, dtype=torch.long))

    if len(filtered_edge_indices) == 0:
        print("No subgraphs found.")
        final_edge_index = torch.empty((2, 0), dtype=torch.long)
        final_subgraph_idx = torch.empty((0,), dtype=torch.long)
        final_edge_labels = torch.empty((0,), dtype=torch.long)
    else:
        final_edge_index = torch.cat(filtered_edge_indices, dim=1)
        final_subgraph_idx = torch.cat(filtered_subgraph_indices, dim=0)
        final_edge_labels = torch.cat(filtered_edge_labels, dim=0)
    return final_edge_index, final_subgraph_idx, final_edge_labels

def build_dgl_graph(x_combined, all_edge_index, all_space_labels, all_relation_type, cdr_to_pep):
    
    assert all_edge_index.size(1) == len(all_space_labels) == len(all_relation_type)
    
    space_and_relation_type_to_idx_dict = {}
    subgraph_idx = []
    edge_labels = []
    edge_label_map = {
        'latent_edge': 0,
        'edge': 1,
        'self_loop_edge': 2,
    }
    
    for space, relation, label in zip(all_space_labels, all_relation_type, all_space_labels):
        if (space, relation) not in space_and_relation_type_to_idx_dict:
            space_and_relation_type_to_idx_dict[(space, relation)] = len(space_and_relation_type_to_idx_dict)
        subgraph_idx.append(space_and_relation_type_to_idx_dict[(space, relation)])
        if label == 0:
            edge_labels.append(edge_label_map['latent_edge'])
        else:
            raise ValueError(f"Unexpected value in all_space_labels: {label}")
   
    final_edge_index, final_subgraph_idx, final_edge_labels = filter_subgraphs_and_concatenate_edges(
        all_edge_index, subgraph_idx, edge_labels, min_num_nodes=10
    )
    
    src, dst = cdr_to_pep
    graph_edge_index = torch.cat([cdr_to_pep, torch.stack([dst, src], dim=0)], dim=1)
    sub_idx_max = final_subgraph_idx.max().item()
    num_new_edges = cdr_to_pep.size(1)
    new_subgraph_idx = torch.cat([
        torch.full((num_new_edges,), fill_value=sub_idx_max + 1, dtype=torch.long),
        torch.full((num_new_edges,), fill_value=sub_idx_max + 2, dtype=torch.long)
    ])
    
    new_edge_labels = torch.full((2 * num_new_edges,), edge_label_map['edge'], dtype=torch.long)
    final_edge_index = torch.cat([final_edge_index, graph_edge_index], dim=1)
    final_subgraph_idx = torch.cat([final_subgraph_idx, new_subgraph_idx])
    final_edge_labels = torch.cat([final_edge_labels, new_edge_labels])
    
    counts = Counter(final_subgraph_idx.tolist())
    for sub_id, count in counts.items():
        print(f"Merged subgraph {sub_id} has {count} edges.")
    
    g = dgl.graph((final_edge_index[0], final_edge_index[1]), num_nodes=x_combined.size(0))
    g.ndata['feat'] = x_combined
    g.edata['subgraph_idx'] = th.tensor(final_subgraph_idx, dtype=th.int64)
    g.edata['edge_label'] = th.tensor(final_edge_labels, dtype=th.int64)

    self_loop_idx = sub_idx_max + 3
    space_and_relation_type_to_idx_dict['self_loop'] = self_loop_idx
    nodes = th.arange(g.num_nodes())
    g.add_edges(
        nodes, nodes,
        data={
            'subgraph_idx': th.full((g.num_nodes(),), self_loop_idx, dtype=th.int64),
            'edge_label': th.full((g.num_nodes(),), edge_label_map['self_loop_edge'], dtype=th.int64),
        }
    )
    return g

def process_data_to_homogeneous_graph(data, edge_type=('cdr3b', 'CBindA', 'peptide'), dim_reduction='umap', consNeighborType='rho'):
    cdr3b_features = data.x_dict[edge_type[0]]
    peptide_features = data.x_dict[edge_type[2]]
    assert cdr3b_features.size(1) == peptide_features.size(1), "Feature dimensions must match."
    x_combined = torch.cat([cdr3b_features, peptide_features], dim=0)
    num_cdr3b_nodes = cdr3b_features.size(0)

    edge_index = data.edge_index_dict[edge_type]
    cdr_to_pep = torch.stack([edge_index[0], edge_index[1] + num_cdr3b_nodes], dim=0)

    assert cdr_to_pep.max() < x_combined.size(0), "Edge index out of range."

    edge_label_index = cdr_to_pep

    src, dst = cdr_to_pep
    reversed_edge_index = torch.stack([dst, src], dim=0)
    graph_edge_index = torch.cat([cdr_to_pep, reversed_edge_index], dim=1)
    graph_edge_index = torch.unique(graph_edge_index, dim=1)

    undirected_g = dgl.graph((graph_edge_index[0], graph_edge_index[1]), num_nodes=x_combined.size(0))
    assert undirected_g.num_nodes() == x_combined.size(0), "Node count mismatch in graph construction."
    
    graph_neighbors, latent_neighbors, threshold_tcr, threshold_pep = compute_structural_neighborhood(undirected_g, 
                                                                                                      x_combined, 
                                                                                                      num_cdr3b_nodes, 
                                                                                                      dim_reduction=dim_reduction, 
                                                                                                      n_components=2, 
                                                                                                      consNeighborType=consNeighborType)
    if consNeighborType == 'rho':
        print(f"Final k_tcr:{threshold_tcr}, k_peptide: {threshold_pep}")
    else:
        print(f"Final rho_tcr:{threshold_tcr}, rho_peptide: {threshold_pep}")                                                                         
    
    combined_edge_index, space_labels = construct_and_concat_edges_with_labels(latent_neighbors)
    assert combined_edge_index.size(1) == space_labels.size(0), "Edge count and label count mismatch."
    
    num_zeros = (space_labels == 0).sum().item()
    # num_zeros = (space_labels == 0).sum().item()
    # print(f"Number of latent edges: {num_ones}, Number of graph edges: {num_zeros}")
    print(f"Number of latent edges: {num_zeros}")
    if dim_reduction == 'umap':
        embeddings = compute_umap_embeddings(x_combined.numpy(), n_components=2)
    elif dim_reduction == 'isomap':
        embeddings = compute_isomap_embeddings(x_combined.numpy(), n_components=2)
    elif dim_reduction == 'pca':
        embeddings = compute_pca_embeddings(x_combined.numpy(), n_components=2)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {dim_reduction}")
    relationships = compute_edge_relationships(combined_edge_index, embeddings)
    
    all_edge_index = combined_edge_index
    all_relation_type_list = relationships
    all_space_labels_list = space_labels.tolist()
    g = build_dgl_graph(x_combined, all_edge_index, all_space_labels_list, all_relation_type_list, cdr_to_pep)
    
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)
    return g, edge_label_index
