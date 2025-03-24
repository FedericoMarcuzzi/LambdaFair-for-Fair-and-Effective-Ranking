from misc import *
import sys
import numpy as np
import argparse

num_threads = 28
dset_n_est = 2000
dataset_path = "../datasets/numpy_datasets/"

parser = argparse.ArgumentParser(description="LambdaFair input parameters.")
parser.add_argument('--dataset', type=str, help='Dataset name.', required=True)
parser.add_argument('--strategy', type=str, help='Strategy name: \'plain\', \'ndcg\', \'rnd\', \'delta\'.', required=True)
parser.add_argument('--ndcg_eval_at', type=int, nargs='+', help='List of cutoff for NDCG evalaution.', required=True)
parser.add_argument('--rnd_eval_at', type=int, nargs='+', help='List of cutoff for rND evalaution.', required=True)
parser.add_argument('--rnd_step', type=int, help='rND metric step.', required=True)
parser.add_argument('--train_cutoff', type=int, help='Training cutoff for both NDCG and rND.', required=True)
parser.add_argument('--alpha', type=float, help='LambdaFair parameters alpha.', required=False, default=1.0)
args = parser.parse_args()

ndcg_out = 'ndcg@' + str(args.ndcg_eval_at)
rnd_out = 'rnd@' + str(args.rnd_step)
str_ndcg_eval_k = "[" + ','.join([str(k) for k in args.ndcg_eval_at]) + "]"
str_rnd_eval_k = "[" + ','.join([str(k) for k in args.rnd_eval_at]) + "]"

output_filename = f"{args.dataset}_trees-{dset_n_est}_ndcgK-{str_ndcg_eval_k}_rndK-{str_rnd_eval_k}_rndStep-{args.rnd_step}_truncLevel-{args.train_cutoff}_alpha-{args.alpha}_strategy-{args.strategy}"
output_path, models_path, results_path, data_path = init_path(data_path=dataset_path, result_path = "", project_name = "lambda_fair")
train_data, train_labels, train_query_lens, valid_data, valid_labels, valid_query_lens, test_data, test_labels, test_query_lens = init_dataset(data_path, args.dataset)
params, feature, threshold = get_params(args.dataset)

upd_params = {
    'lambda_fair' : args.strategy,
    'num_threads' : num_threads,
    'metric': ['ndcg', 'rnd'],
    'ndcg_eval_at': args.ndcg_eval_at,
    'rnd_eval_at' : args.rnd_eval_at,
    'alpha_lambdafair': args.alpha,
    'rnd_step': args.rnd_step,
    'lambdarank_truncation_level' : args.train_cutoff,
    'lambdarank_norm' : True,
    'group_labels' : get_group_labels(train_data, feature, threshold), # get protected group labels for training
}

# {"eval_at_rND", {"rnd_eval_at", "rnd_at"}},

params.update(upd_params)

# get protected group labels for evaluation
eval_group_labels = [
    get_group_labels(train_data, feature, threshold), 
    get_group_labels(valid_data, feature, threshold), 
    get_group_labels(test_data, feature, threshold)
]

# removes protected features
train_data = np.delete(train_data, feature, 1)
valid_data = np.delete(valid_data, feature, 1)
test_data = np.delete(test_data, feature, 1)

if "mslr" in params:
    train_data = np.delete(train_data, feature-1, 1) # QualityScore
    valid_data = np.delete(valid_data, feature-1, 1) # QualityScore
    test_data = np.delete(test_data, feature-1, 1) # QualityScore

# preprare datasets in lightGBM format
eval_set = [(train_data, train_labels), (valid_data, valid_labels), (test_data, test_labels)]
eval_group = [train_query_lens, valid_query_lens, test_query_lens]
eval_names = ['train', 'valid', 'test']
data_sets, data_names = prepare_lightgbm_dataset((train_data, train_labels, train_query_lens), eval_set, eval_group, eval_names, eval_group_labels=eval_group_labels)

# evaluation over iteration
results = {}
# model training
model = lgb.train(params, data_sets[0], num_boost_round = dset_n_est, keep_training_booster=True, valid_sets = data_sets, valid_names = data_names, callbacks=[lgb.log_evaluation(), lgb.record_evaluation(results)])
# save model
model.save_model(models_path + output_filename + '.txt')
# save results
dump_obj(results, results_path, output_filename)