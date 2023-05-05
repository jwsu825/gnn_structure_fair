import networkx as nx

def extract_node_id_for_label(node_label,label):
  node_list = []
  for id in range(len(node_label)):
    if node_labels[id] == label:
      node_list.append(id)
  return node_list

def extract_node_id_from_mask(mask):
  node_list = []
  for id in range(len(mask)):
    if mask[id] == True:
      node_list.append(id)
  return node_list

def extract_index_from_mask(mask):
  return [idx for idx, v in enumerate(mask) if v]

def node_id_list_to_mask(node_id_list, num_node):
  i = 0
  bool_list = num_node * [False]
  for i in range(num_node):
    if i in node_id_list:
      bool_list[i] = True
  return bool_list

def node_id_in_degree_sorted(node_id,graph):
  degree_list  = list(graph.degree(node_id))
  degree_list_sort = sorted(degree_list,reverse = True, key=lambda element: element[1])
  node_list_sort = [ele[0] for ele in degree_list_sort]
  return node_list_sort

def degree_distribution(graph, node_id_list):  
  degree_vertex_dict = {}
  vertex_degree_dict = dict(graph.degree(node_id_list))
  for k, v in vertex_degree_dict.items():
      degree_vertex_dict[v] = degree_vertex_dict.get(v, []) + [k]
  return degree_vertex_dict

def degree_dict(graph):
  vertex_list = list(graph.nodes)
  degree_dict = dict(graph.degree(vertex_list))
  return degree_dict

def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return list(nbrs)

def extract_k_hop_neighbor(graph,vertex,k):
  neighbor_list = []
  visited       = set()
  neighbor_list.append(set(vertex))
  visited.union(set(vertex))
  for i in range(k):
    neighbor_candidate = set()
    for v in neighbor_list[i]:
      to_vist = set([n for n in graph.neighbors(v)])
      neighbor_candidate.update(to_vist)
    next_neigbhor = set()
    for u in neighbor_candidate:
      if u not in visited:
        next_neigbhor.add(u)
        visited.add(u)
    neighbor_list.append(next_neigbhor)
  return neighbor_list


def extract_k_hop_neighbor_label(graph,vertex,k,node_label_list):
  neighbor_list = []
  neighbor_list_label = []
  visited       = set()
  neighbor_list.append({vertex})
  neighbor_list_label.append([node_label_list[vertex]])
  visited.add(vertex)

  for i in range(k):
    neighbor_candidate = set()
    for v in neighbor_list[i]:
      to_vist = set([n for n in graph.neighbors(v)])
      neighbor_candidate.update(to_vist)
    next_neigbhor       = set()
    next_neigbhor_label = []
    for u in neighbor_candidate:
      if u not in visited:
        next_neigbhor.add(u)
        next_neigbhor_label.append(node_label_list[u])
        visited.add(u)
    neighbor_list.append(next_neigbhor)
    neighbor_list_label.append(next_neigbhor_label)
  return neighbor_list_label


def l2diff(x1, x2):
    return (x1-x2).norm(p=2)

def moment_diff(sx1, sx2, k):
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    return l2diff(ss1,ss2)


def measure_homophily(graph):
  labels = graph.ndata['label']
  nxg = graph.to_networkx().to_undirected()
  edges_list = list(nxg.edges())
  total_edge_num = len(edges_list)
  homo_edge_num = 0
  for edge in edges_list:
    v1_id = edge[0]
    v2_id = edge[1]
    v1_label = labels[v1_id]
    v2_label = labels[v2_id]
    if v1_label == v2_label:
      homo_edge_num = homo_edge_num + 1

  return homo_edge_num/total_edge_num

