import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, TransformerConv, FiLMConv
from collections import Counter


class GeomGCNSingleChannel(nn.Module):
    def __init__(self, in_feats, out_feats, num_divisions, activation=F.leaky_relu, 
                 dropout_prob=0.2, merge='sum', latent_weight = 0.1, selfloop_weight = 0):
        super(GeomGCNSingleChannel, self).__init__()
        self.num_divisions = num_divisions
        self.in_feats_dropout = nn.Dropout(dropout_prob)
        self.activation = activation
        self.out_feats = out_feats
        self.edge_types = [f"edge_type_{i}" for i in range(self.num_divisions)]
        self.merge = merge
        self.latent_weight = latent_weight
        self.selfloop_weight = selfloop_weight
        self.convs = nn.ModuleDict({
            edge_type: SAGEConv((-1, -1), self.out_feats)
            for edge_type in self.edge_types
        })
    def get_subgraphs(self, g, num_divisions=None, edge_types=None):
        if 'subgraph_idx' not in g.edata:
            raise ValueError("Graph `g` must have `subgraph_idx` in edata.")
        if 'edge_label' not in g.edata:
            raise ValueError("Graph `g` must have `edge_label` in edata.")
        subgraph_idx = g.edata['subgraph_idx']
        edge_labels = g.edata['edge_label']
        if num_divisions is None:
            num_divisions = self.num_divisions
        if edge_types is None:
            edge_types = self.edge_types
        src, dst = g.edges()
        edge_index_dict = {}
        edge_label_stats = {}
        for idx in range(num_divisions):
            mask = (subgraph_idx == idx)
            edge_indices = torch.nonzero(mask, as_tuple=True)[0]
            edge_index_dict[edge_types[idx]] = torch.stack([src[edge_indices], dst[edge_indices]], dim=0)
            subgraph_edge_labels = edge_labels[mask].tolist()
            edge_label_counts = Counter(subgraph_edge_labels)
            edge_label_stats[edge_types[idx]] = edge_label_counts
        return edge_index_dict, edge_label_stats
    def forward(self, g, feature):
        if feature.size(0) != g.number_of_nodes():
            raise ValueError("Feature and graph node count mismatch.")
        if 'norm' not in g.ndata:
            raise ValueError("Graph `g` must have `norm` in ndata.")
        feature = self.in_feats_dropout(feature)
        edge_index_dict, edge_label_stats = self.get_subgraphs(g, self.num_divisions, self.edge_types)
        hidden_list = []
        weights = {
            0: self.latent_weight,
            1:1,
            2: self.selfloop_weight
        }
        for edge_type, edge_index in edge_index_dict.items():
            stats = edge_label_stats[edge_type]
            assert len(stats) == 1, f"Subgraph {edge_type} contains more than one edge type."
            label = next(iter(stats.keys()))
            weight = weights[label]

            h = self.convs[edge_type](feature, edge_index)
            h = self.activation(h)

            h = h * weight
            hidden_list.append(h)

        x = torch.stack(hidden_list, dim=0)
        if self.merge=='sum':
            x = x.sum(dim=0)
        else:
            x = x.mean(dim=0)
        return x


class GeomGCN(nn.Module):
    def __init__(self, in_feats, out_feats, num_divisions, activation, 
                 num_heads, dropout_prob, ggcn_merge, channel_merge,
                 latent_weight, selfloop_weight):
        super(GeomGCN, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(
                GeomGCNSingleChannel(in_feats, out_feats,num_divisions, activation, dropout_prob,
                                     ggcn_merge, latent_weight, selfloop_weight))
        self.channel_merge = channel_merge
    def forward(self, g, feature):
        all_attention_head_outputs = []
        for head in self.attention_heads:
            out = head(g, feature)
            all_attention_head_outputs.append(out)
        if self.channel_merge == 'cat':
            return th.cat(all_attention_head_outputs, dim=1)
        else:
            return th.mean(th.stack(all_attention_head_outputs, dim=0), dim=0)


class GeomGCNNet(nn.Module):
    def __init__(self, in_feats, num_hidden, out_feats, num_divisions, num_heads_layer_one, 
                 dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge, latent_weight, selfloop_weight):
        super(GeomGCNNet, self).__init__()
        self.geomgcn1 = GeomGCN(
            in_feats=in_feats,
            out_feats=num_hidden,
            num_divisions = num_divisions,
            activation=F.leaky_relu,
            num_heads=num_heads_layer_one,
            dropout_prob=dropout_rate,
            ggcn_merge = layer_one_ggcn_merge,
            channel_merge=layer_one_channel_merge,
            latent_weight = latent_weight,
            selfloop_weight = selfloop_weight
        )
        if layer_one_ggcn_merge == 'cat':
            layer_one_ggcn_merge_multiplier = num_divisions
        else:
            layer_one_ggcn_merge_multiplier = 1
        if layer_one_channel_merge == 'cat':
            layer_one_channel_merge_multiplier = num_heads_layer_one
        else:
            layer_one_channel_merge_multiplier = 1
    def forward(self, g, features):
        x = self.geomgcn1(g, features)
        return x


class MLP(torch.nn.Module):
    def __init__(self, mode, hidden_channels=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lin1 = None
        self.lin2 = nn.Linear(512, 256)
        self.mode = mode
        if self.mode=="binary":
            self.lin3 = nn.Linear(256, 1)
        else:
            self.lin3 = nn.Linear(256, 3)
    def forward(self, node_features, edge_label_index):
        if self.lin1 is None:
            input_dim = node_features.size(1) * 2
            self.lin1 = nn.Linear(input_dim, 512).to(node_features.device)
        row, col = edge_label_index # cdr-pep
        x = torch.cat([node_features[row], node_features[col]], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        intermediate_x = x
        x = self.lin3(x)
        if self.mode=="binary":
            return x.view(-1), intermediate_x
        elif self.mode=="trinary":
            return x, intermediate_x
        else:
            raise ValueError("mode must be'binary' or 'trinary'")
    


class TridentTCR(torch.nn.Module):
    def __init__(self, mode, num_input_features, num_hidden, num_output_classes, num_divisions, num_heads_layer_one, 
                 dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge, latent_weight, selfloop_weight):
        super(TridentTCR, self).__init__()
        self.encoder = GeomGCNNet(
            in_feats=num_input_features,
            num_hidden=num_hidden,
            out_feats=num_output_classes,
            num_divisions = num_divisions,
            num_heads_layer_one=num_heads_layer_one,
            dropout_rate=dropout_rate,
            layer_one_ggcn_merge = layer_one_ggcn_merge,
            layer_one_channel_merge=layer_one_channel_merge,
            latent_weight=latent_weight,
            selfloop_weight = selfloop_weight
        )
        self.decoder = MLP(mode, num_output_classes)
    def forward(self, g, features, edge_label_index):
        z = self.encoder(g, features)
        return self.decoder(z, edge_label_index)

