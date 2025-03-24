import re
import os

import numpy as np
import lightgbm as lgb

from pathlib import Path
from itertools import groupby
from pickle import dump


def get_params(param_text):
    if "german_age" in param_text or "german_sex" in param_text:
        params = {
            'learning_rate' : 0.01,
            'num_leaves' : 32,
            'min_data_in_leaf' : 5,
            'max_bin' : 255,

            'objective':  'lambdarank',
            'boosting' : 'gbdt',
            'lambdarank_norm' : True,
            'seed' : 7
        }

        return params, 24, 1
    
    if param_text == "mslr-web30k-fold1":
        params = {
            'learning_rate' : 0.01,
            'num_leaves' : 400,
            'min_data_in_leaf' : 50,
            'max_bin' : 255,
            'min_sum_hessian_in_leaf' : 0,

            'objective':  'lambdarank',
            'boosting' : 'gbdt',
            'lambdarank_norm' : False,
            'seed' : 7
        }

        return params, 132, 10
    
    if param_text == "hmdaCT_sex":
        params = {
            'learning_rate' : 0.05,
            'num_leaves' : 64,
            'min_data_in_leaf' : 5,
            'max_bin' : 255,

            'objective':  'lambdarank',
            'boosting' : 'gbdt',
            'lambdarank_norm' : True,
            'seed' : 7
        }

        return params, 817, 1

def init_path(data_path='../../datasets/numpy_datasets/', result_path = '../', project_name = 'ALL'):
    output_path = os.path.join(result_path, 'output/')
    models_path = os.path.join(output_path, 'models_' + project_name + '/')
    results_path = os.path.join(output_path, 'results_'+ project_name + '/')

    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(models_path).mkdir(parents=True, exist_ok=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)
    return output_path, models_path, results_path, data_path

def init_dataset(data_path, param_text):
    print('[INFO] Loading data:')
    train_data, train_labels, train_query_lens = load_data_npy(data_path + param_text + '_train.npy')
    print('[INFO] Training set loaded:')
    valid_data, valid_labels, valid_query_lens = load_data_npy(data_path + param_text + '_valid.npy')
    print('[INFO] Validation set loaded:')
    test_data, test_labels, test_query_lens = load_data_npy(data_path + param_text + '_test.npy')
    print('[INFO] Testing set loaded:')
    return train_data, train_labels, train_query_lens, valid_data, valid_labels, valid_query_lens, test_data, test_labels, test_query_lens

def prepare_lightgbm_dataset(train_set, eval_set, eval_group, eval_names, eval_group_labels):
    lightgbm_set = []
    train_set = lgb.Dataset(train_set[0], label=train_set[1], group=train_set[2], params={'eval_group_labels' : eval_group_labels[0]})
    lightgbm_set = [train_set]

    # loads validation and test set
    for i, data in enumerate(eval_set):
        ds = lgb.Dataset(data[0], data[1], group=eval_group[i], reference=train_set, params={'eval_group_labels' : eval_group_labels[i]})
        lightgbm_set.append(ds)

    new_eval_names = ['train_set'] + eval_names  
    return lightgbm_set, new_eval_names

def load_data_npy(filename):
    X = np.load(filename)

    data = X[:,:-2]
    labels = X[:,-2]
    query_lens = np.asarray([len(list(group)) for key, group in groupby(X[:,-1])])
    return data, labels.astype(int), query_lens.astype(int)

def get_group_labels(data, feature, threshold):
    return (data[:, feature] < threshold).astype(int)

def dump_obj(obj, path, filename):
    dump(obj, open(os.path.join(path, filename + '.pkl'), 'wb'))