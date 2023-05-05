import data
import sampling
import train
from model import GCN, SAGE, GAT, GCN2

#py library
import torch
import os
import random
import argparse
import math
import os
import dgl
import statistics 

from typing import Tuple, Optional
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from tqdm import tqdm

import dgl.nn as dglnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd
import utility

inv_size = 5 

def subgroup_embedding(model_name, dataset, args):
  graph = data.load_data(dataset)
  nx_graph = graph.to_networkx().to_undirected()
  graph_node_id = list(nx_graph.nodes)
  node_size = len(graph_node_id)

  args['graph'] = graph
  args['num_infeat'] = graph.ndata['feat'].shape[1]
  args['num_class'] = int(graph.ndata['label'].max().item() + 1)

  train_mask = graph.ndata['train_mask']
  args['train_mask'] = train_mask

  train_set = utility.extract_index_from_mask(train_mask)
  test_node_id = [i for i in graph_node_id if not i in train_set]
  test_mask    = utility.node_id_list_to_mask(test_node_id,node_size)
  args['test_mask'] = test_mask

  training_info = {}
  if model_name == 'GCN':
    model = GCN(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
  if model_name == 'SAGE':
    model = SAGE(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
  if model_name == 'GAT':
    model = GAT(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
  if model_name == 'GCN2':
    model = GCN2(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])

  train_loss,test_acc,best_acct, model = train.train(args,model)
  
  training_info['train_loss'] = train_loss
  training_info['test_acc']   = test_acc
  training_info['best_acct']  = best_acct

  subgroup_embedding_dist = []

  model.eval()
  representation_list = model(args['graph'],args['graph'].ndata['feat'])
  predictions = F.log_softmax(representation_list, 1)
  trainset_prediction = predictions[args['train_mask']] 
  _, predicted_label = torch.max(trainset_prediction, dim=1)
  transet_label       =args['graph'].ndata['label'][args['train_mask']]

  well_trained_set = []
  for index in range(len(predicted_label)):
    if predicted_label[index] == transet_label[index]:
      well_trained_set.append(train_set[index])

  well_train_mask = utility.node_id_list_to_mask(well_trained_set,node_size)
  train_embedding = representation_list[well_train_mask]

  neighbor_list = utility.extract_k_hop_neighbor(nx_graph,well_trained_set,inv_size)

  for i in tqdm(range(inv_size)):
    group_node = neighbor_list[i]
    group_test_mask = utility.node_id_list_to_mask(group_node,node_size)
    subgroup_embedding = representation_list[group_test_mask]
    embeddings_dist = meanDff(subgroup_embedding,train_embedding)
    subgroup_embedding_dist.append(embeddings_dist.item())

  return subgroup_embedding_dist      


def meanDff(x1, x2):
  return (x1.mean(0)-x2.mean(0)).norm(p=2)
