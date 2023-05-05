import dgl
import numpy
import torch
import utility
import sampling

DEFAULT_TRAIN_PERCENT = 0.1

def load_data(name):

  if name == 'cora':
    dataset = dgl.data.CoraGraphDataset()
    graph = dataset[0]
  if name == 'cite':
    dataset = dgl.data.CiteseerGraphDataset()
    graph = dataset[0]
  if name == 'pubmed':
    dataset = dgl.data.PubmedGraphDataset()
    graph = dataset[0]
  if name == 'cora_full':
    dataset= dgl.data.CoraFullDataset()
    graph = dataset[0]
    node_num = len(graph.ndata['label'])
    node_id_list = list(range(node_num))
    train_size = int(DEFAULT_TRAIN_PERCENT*node_num)
    train_set,test_set = sampling.uniform_sampling(node_id_list,train_size)
    graph.ndata['train_mask'] = torch.BoolTensor(utility.node_id_list_to_mask(train_set,node_num))
    graph.ndata['test_mask']  = torch.BoolTensor(utility.node_id_list_to_mask(test_set,node_num))

  if name == 'coau_cs':
    dataset = dgl.data.CoauthorCSDataset()
    graph = dataset[0]
    node_num = len(graph.ndata['label'])
    node_id_list = list(range(node_num))
    train_size = int(DEFAULT_TRAIN_PERCENT*node_num)
    train_set,test_set = sampling.uniform_sampling(node_id_list,train_size)
    graph.ndata['train_mask'] = torch.BoolTensor(utility.node_id_list_to_mask(train_set,node_num))
    graph.ndata['test_mask']  = torch.BoolTensor(utility.node_id_list_to_mask(test_set,node_num))

  if name == 'coau_phy':
    dataset = dgl.data.CoauthorPhysicsDataset()
    graph = dataset[0]
    node_num = len(graph.ndata['label'])
    node_id_list = list(range(node_num))
    train_size = int(DEFAULT_TRAIN_PERCENT*node_num)
    train_set,test_set = sampling.uniform_sampling(node_id_list,train_size)
    graph.ndata['train_mask'] = torch.BoolTensor(utility.node_id_list_to_mask(train_set,node_num))
    graph.ndata['test_mask']  = torch.BoolTensor(utility.node_id_list_to_mask(test_set,node_num))

  if name == 'ogbn-arxiv':
    dataset = DglNodePropPredDataset('ogbn-arxiv')
    graph, node_labels = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]

    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']



  return graph

