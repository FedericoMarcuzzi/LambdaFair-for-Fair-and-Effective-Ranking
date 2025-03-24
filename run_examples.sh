#!/bin/sh
python3 run.py --dataset german_age --strategy plain --ndcg_eval_at 15 --rnd_eval_at 15 --rnd_step 5 --train_cutoff 20 # LambdaMart
python3 run.py --dataset german_age --strategy rnd --ndcg_eval_at 15 --rnd_eval_at 15 --rnd_step 5 --train_cutoff 20 --alpha 0.9 # LambdaFair rND+
python3 run.py --dataset german_age --strategy delta --ndcg_eval_at 15 --rnd_eval_at 15 --rnd_step 5 --train_cutoff 20 --alpha 0.9 # LambdaFair Delta_rND
python3 run.py --dataset german_age --strategy ndcg --ndcg_eval_at 15 --rnd_eval_at 15 --rnd_step 5 --train_cutoff 20 --alpha 0.9 # LambdaFair NDCG+