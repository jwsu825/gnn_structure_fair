import subgroup_performance_exp
import subgroup_dist_exp
import label_exp
import statistics
import sampling
import random
import numpy as np
import torch

import matplotlib.pyplot as plt

global args 

args = {
   "lr" : 1e-2,
   "weight_decay" : 5e-4,
   "dropout" : 0,
   "epoches" : 60,
   "num_layer" : 2,
   "num_hidfeat" : 32,
   "learning_set_size":10,
   "trial_num" :10,
   "loss":'cross_entropy',
   "coverage"  : 1
}


learning_set_size = 10
trial_num  = 10
coverage  = 1


if __name__ == '__main__':

    #subgroup performance
    subgroup_performance_exp.subgroup_performance(args,'cora','GCN')

    #subgroup distance
    subgroup_dist_exp.subgroup_performance(args,'cora','GCN')

    #label experiment
    label_exp.max_cover_exp('GCN','cora',args)
