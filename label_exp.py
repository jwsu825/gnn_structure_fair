import data
import sampling
import train


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
from model import GCN, SAGE, GAT, GCN2

import dgl.nn as dglnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd

def uniform_sampling_exp(model_name,dataset,args):
  trials_num = args['trial_num']
  trainset_size = args['learning_set_size']
  graph = data.load_data(dataset)

  args['graph'] = graph
  args['num_infeat'] = graph.ndata['feat'].shape[1]
  args['num_class'] = int(graph.ndata['label'].max().item() + 1)

  nx_G = graph.to_networkx().to_undirected()
  connected_comps = sorted(nx.connected_components(nx_G), key = len, reverse=True)
  data_population = list(connected_comps[0])

  test_acc_list = []
  for i in tqdm(range(trials_num)):
    if model_name == 'GCN':
      model = GCN(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
    if model_name == 'SAGE':
      model = SAGE(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
    if model_name == 'GAT':
      model = GAT(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
    if model_name == 'GCN2':
      model = GCN2(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])

    train_mask,test_mask = sampling.uniform_sampling(data_population,trainset_size)
    args['train_mask'] = train_mask
    args['test_mask'] = test_mask
    train_loss,test_acc,best_acct, model = train.train(args,model)
    test_acc_list.append(best_acct)
  return test_acc_list

def importance_base_exp(model_name,dataset,args, method):
  trials_num = args['trial_num']
  trainset_size = args['learning_set_size']
  graph = data.load_data(dataset)

  args['graph'] = graph
  args['num_infeat'] = graph.ndata['feat'].shape[1]
  args['num_class'] = int(graph.ndata['label'].max().item() + 1)

  nx_G = graph.to_networkx().to_undirected()
  connected_comps = sorted(nx.connected_components(nx_G), key = len, reverse=True)
  data_population = list(connected_comps[0])
  comp_graph = nx_G.subgraph(data_population)

  if method == 'degree':
    degree_dict = list(nx_G.degree(data_population))
    weight_list = [ele[1] for ele in degree_dict]
  if method == 'cluster':
    cluster_dict = nx.square_clustering(comp_graph)
    weight_list = [ele[1] for ele in cluster_dict.items()]
  if method == 'center':
    center_dict = nx.degree_centrality(comp_graph)
    weight_list = [ele[1] for ele in center_dict.items()]

  test_acc_list = []
  for i in tqdm(range(trials_num)):
    if model_name == 'GCN':
      model = GCN(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
    if model_name == 'SAGE':
      model = SAGE(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
    if model_name == 'GAT':
      model = GAT(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
    if model_name == 'GCN2':
      model = GCN2(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])

    train_mask,test_mask = sampling.importance_base_sampling(data_population,weight_list,trainset_size)
    args['train_mask'] = train_mask
    args['test_mask'] = test_mask
    train_loss,test_acc,best_acct, model = train.train(args,model)
    test_acc_list.append(best_acct)

  return test_acc_list

def max_cover_exp(model_name,dataset,args,not_remove=False):

  trials_num = args['trial_num']
  trainset_size = args['learning_set_size']
  coverage  = args['coverage']

  graph = data.load_data(dataset)
  args['graph'] = graph
  args['num_infeat'] = graph.ndata['feat'].shape[1]
  args['num_class'] = int(graph.ndata['label'].max().item() + 1)

  nx_G = graph.to_networkx().to_undirected()
  connected_comps = sorted(nx.connected_components(nx_G), key = len, reverse=True)
  data_population = list(connected_comps[0])

  largest_comp = nx_G.subgraph(connected_comps[0])
  test_acc_list = []
  for i in tqdm(range(trials_num)):
    if model_name == 'GCN':
      model = GCN(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
    if model_name == 'SAGE':
      model = SAGE(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
    if model_name == 'GAT':
      model = GAT(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])
    if model_name == 'GCN2':
      model = GCN2(in_feats = args['num_infeat'], hid_feats = args['num_hidfeat'], out_feats = args['num_class'], num_layers = args['num_layer'])

    if not_remove == True:
      train_mask,test_mask = sampling.max_cover_nr(largest_comp,trainset_size,coverage)
    if not_remove == False:
      train_mask,test_mask = sampling.max_cover(largest_comp,trainset_size,coverage)

    args['train_mask'] = train_mask
    args['test_mask'] = test_mask

    train_loss,test_acc,best_acct, model = train.train(args,model)
    test_acc_list.append(best_acct)

  return test_acc_list

