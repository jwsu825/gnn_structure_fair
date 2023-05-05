import torch
import dgl

from typing import Tuple, Optional
from torch import nn
from torch import Tensor
from torch.nn import Parameter

import dgl.nn as dglnn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers_GCN = nn.ModuleList([])
        if num_layers == 1:
            self.layers_GCN.append(dglnn.GraphConv(in_feats, out_feats))
        elif num_layers == 2:
            self.layers_GCN.append(dglnn.GraphConv(in_feats, hid_feats))
            self.layers_GCN.append(dglnn.GraphConv(hid_feats, out_feats))
        else:
            self.layers_GCN.append(dglnn.GraphConv(in_feats, hid_feats))
            for _ in range(num_layers-2):
                self.layers_GCN.append(dglnn.GraphConv(hid_feats, hid_feats))
            self.layers_GCN.append(dglnn.GraphConv(hid_feats, out_feats))

    def forward(self, graph, inputs):
        graph = dgl.add_self_loop(graph)
        x = inputs
        for i in range(self.num_layers-1):
          x = self.layers_GCN[i](graph,x)
          x = F.relu(x)
        x = self.layers_GCN[self.num_layers-1](graph, x)
        return x


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        if num_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_feats, out_feats,aggregator_type='mean'))
        elif num_layers == 2:
            self.layers.append(dglnn.SAGEConv(in_feats, hid_feats,aggregator_type='mean'))
            self.layers.append(dglnn.SAGEConv(hid_feats, out_feats,aggregator_type='mean'))    
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, hid_feats,aggregator_type='mean'))
            for _ in range(num_layers-2):
                self.layers.append(dglnn.SAGEConv(hid_feats, hid_feats,aggregator_type='mean'))
            self.layers.append(dglnn.SAGEConv(hid_feats, out_feats,aggregator_type='mean'))


    def forward(self, graph, inputs):
        graph = dgl.add_self_loop(graph)
        x = inputs
        for i in range(self.num_layers-1):
          x = self.layers[i](graph,x)
          x = F.relu(x)
        x = self.layers[self.num_layers-1](graph,x)
        return x


class GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        num_heads = 2
        if num_layers == 1:
            self.layers.append(dglnn.GATConv(in_feats, out_feat,num_heads))
        elif num_layers == 2:
            self.layers.append(dglnn.GATConv(in_feats, hid_feats,num_heads))
            self.layers.append(dglnn.GATConv(hid_feats*num_heads, out_feats,num_heads))
        else:
            self.layers.append(dglnn.GATConv(in_feats, hid_feats,num_heads))
            for _ in range(num_layers-2):
                self.layers.append(dglnn.GATConv(hid_feats*num_heads, hid_feats,num_heads))
            self.layers.append(dglnn.GATConv(hid_feats *num_heads, out_feats,num_heads))


    def forward(self, graph, inputs):
        graph = dgl.add_self_loop(graph)
        x = inputs
        for i in range(self.num_layers-1):
          x = self.layers[i](graph,x)
          x = F.relu(x)
          x = torch.flatten(x, start_dim=1)
        x = self.layers[self.num_layers-1](graph,x)
        x = torch.flatten(x, start_dim=1)
        return x


class GCN2(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers_GCN = nn.ModuleList([])
        self.encoder    = nn.Linear(in_feats, hid_feats)
        self.decoder    = nn.Linear(hid_feats, out_feats)
        for i in range(num_layers):
            self.layers_GCN.append(dglnn.GCN2Conv(hid_feats, layer=i+1, alpha=0.5,project_initial_features=True, allow_zero_in_degree=True))
    
    def forward(self, graph, inputs):
        graph = dgl.add_self_loop(graph)
        x = inputs
        x = self.encoder(x)
        x = F.relu(x)
        res = x
        for i in range(self.num_layers):
          res = self.layers_GCN[i](graph, res, x)
        x = self.decoder(res)
        x = F.relu(x)
        return x