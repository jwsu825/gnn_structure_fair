import utility
import random
import networkx as nx

def importance_base_sampling(population,weight_seq,size):
  train_set = []
  new_population = population
  new_deg_seq    = weight_seq
  for i in range(size):
    data = random.choices(new_population, new_deg_seq, k=1)[0]
    train_set.append(data)
    index = new_population.index(data)
    new_population.pop(index)
    new_deg_seq.pop(index)
  test_set  = [i for i in population if not i in train_set]
  return train_set, test_set

def uniform_sampling(population,size):
  train_set = random.sample(population, k=size)
  test_set  = [i for i in population if not i in train_set]
  return train_set, test_set

def weighted_vertex_cover(graph, weighted_vertex_cover):
  cover = nx.algorithms.approximation.min_weighted_vertex_cover(graph,None)
  return cover

def k_center(graph,k,h):
  train_set = []
  original_population = list(graph.nodes)
  current_graph = nx.Graph(graph)

  for i in range(k):
    population = list(current_graph.nodes)
    degree_dict = list(current_graph.degree)
    degree_list = [ele[1] for ele in degree_dict]
    data = random.choices(population, degree_list, k=1)[0]
    train_set.append(data)
    to_remove = utility.knbrs(current_graph, data, h)  
    current_graph.remove_nodes_from(to_remove)

  test_set  = [i for i in original_population if not i in train_set]
  return train_set,test_set


def max_cover(graph,k,h):
  train_set = []
  original_population = list(graph.nodes)
  current_graph = nx.Graph(graph)

  for i in range(k):
    population = list(current_graph.nodes)
    degree_dict = list(current_graph.degree)
    degree_list = [ele[1] for ele in degree_dict]
    max_deg = max(degree_list)
    max_node_index = [i for i, e in enumerate(degree_list) if e == max_deg]
    chosen_node_index = random.choices(max_node_index, k=1)[0]
    data = population[chosen_node_index]
    train_set.append(data)
    to_remove = utility.knbrs(current_graph, data, h)  
    current_graph.remove_nodes_from(to_remove)

  test_set  = [i for i in original_population if not i in train_set]
  return train_set,test_set


def set_cover_formulation(graph,k,h):
  cover_list = []
  node_set = set(graph.nodes)
  for v in node_set:
    k_neighbor_set = set(utility.knbrs(graph, v, h))  
    cover_list.append(k_neighbor_set)

  uncovered_elem_num_list = [len(i) for i in cover_list]
  max_uncover_num = max(uncovered_elem_num_list)
  covered_element = set()
  num_set = 0

  while max_uncover_num > 0:
    num_set = num_set + 1
    max_uncover_set_index = uncovered_elem_num_list.index(max_uncover_num)
    max_uncover_set       = cover_list[max_uncover_set_index]
    covered_element.union(max_uncover_set)

    for i in range(len(cover_list)):
      cover_list[i] = cover_list[i].difference(covered_element) 

    uncovered_elem_num_list = [len(i) for i in cover_list]
    max_uncover_num = max(uncovered_elem_num_list)

  return num_set

def max_cover_nr(graph,k,h):
  train_set = []
  cover_set_list = []
  node_set = set(graph.nodes)
  for v in node_set:
    k_neighbor_set = set(utility.knbrs(graph, v, h))  
    cover_set_list.append(k_neighbor_set)

  population = list(graph.nodes)
  degree_dict = list(graph.degree)
  cover_index_list = [ele[1] for ele in degree_dict]
  covered_vertex_set = set()

  for i in range(k):
    max_cover = max(cover_index_list)
    max_node_index = [i for i, e in enumerate(cover_index_list) if e == max_cover]
    chosen_node_index = random.choices(max_node_index, k=1)[0]
    data = population[chosen_node_index]
    train_set.append(data)
    covered_vertex_set.update(utility.knbrs(graph, data, h))

    for j in range(len(cover_index_list)):
      cover_index_list[j] = len(cover_set_list[j].difference(covered_vertex_set))
    
  test_set  = [i for i in population if not i in train_set]
  return train_set,test_set