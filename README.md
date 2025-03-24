LambdaFair-for-Fair-and-Effective-Ranking
===============================

This code is for ECIR 2025 full paper [LambdaFair for Fair and Effective Ranking]().

Abstract
---
Traditional machine learning algorithms are known to amplify bias in data or introduce new biases during the learning process, often resulting in discriminatory outcomes that impact individuals from marginalized or underrepresented groups.
In information retrieval, one application of machine learning is learning-to-rank frameworks, typically employed to reorder items based on their relevance to user interests. This focus on effectiveness can lead to rankings that unevenly distribute exposure among groups, affecting their visibility to the final user.
Consequently, ensuring fair treatment of protected groups has become a pivotal challenge in information retrieval to prevent discrimination, alongside the need to maximize ranking effectiveness.
This work introduces LambdaFair, a novel in-processing method designed to jointly optimize effectiveness and fairness ranking metrics.
LambdaFair builds upon the LambdaMART algorithm, harnessing its ability to train highly effective models through additive ensembles of decision trees while integrating fairness awareness.
We evaluate LambdaFair on three publicly available datasets, comparing its performance with state-of-the-art learning algorithms in terms of both fairness and effectiveness.
Our experiments demonstrate that, on average, LambdaFair achieves 6.7\% higher effectiveness and only 0.4\% lower fairness compared to state-of-the-art fairness-oriented learning algorithms.
This highlights LambdaFair’s ability to improve fairness without sacrificing the model's effectiveness.

Implementation
---
**LambdaFair** is a learning-to-rank strategy for fair and effective ranking, built on top of [LightGBM](https://github.com/microsoft/LightGBM).
The code also implements the [rND](https://dl.acm.org/doi/10.1145/3085504.3085526) metric, accessible through the parameter ``metric`` e.g., ``metric=['ndcg', 'rnd']``, for evaluating both NDCG and rND.

Usage
---
**LambdaFair** is accessible through the ``lambdarank`` parameter ``lambda_fair`` (or ``lambdafair``) with the following value:
  - ``"plain"`` to enforce the original LambdaMART algorithm (default).
  - ``"rnd"`` to enforce LambdaFair with the rND+ strategy (Fairness driven).
  - ``"ndcg"`` to enforce LambdaFair with the NDCG+ strategy (Effectiveness driven).
  - ``"delta"`` to enforce LambdaFair with the ΔrND strategy (Variation on swap).

Parameters
---
LambdaFair use the folowwing parameters to works
  - ``rnd_eval_at`` similar to ``ndcg_eval_at`` it allow to specify the rND metric evalaution cutoff.
  - ``alpha_lambdafair`` is the weight representing the relative importance of the two metrics, rND and NDCG, in the convex combination. The parameter is defined in [0,1].
  - ``rnd_step`` specifies the rND bin size in the training and evaluation process.
  - ``group_labels`` is a vector of integers encoding the document group membership labels. The labels must be encoded with integer ``0`` for non-protected and ``1`` for protected.
  - ``eval_group_labels`` is a vector of integers encoding the document group membership labels for the evaluation sets. The LightGBM **Dataset** class has a field in the parameters ``params``, named ``eval_group_labels``, which, like `group_labels`, encodes ``0`` for non-protected and ``1`` for protected.

Installation
---
Follow the [installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) as mentioned in the LightGBM GitHub [repository](https://github.com/microsoft/LightGBM).
Where needed, replace the repository ``https://github.com/microsoft/LightGBM`` with this one.

Citation
---

```
```
