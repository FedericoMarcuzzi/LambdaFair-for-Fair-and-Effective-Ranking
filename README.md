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

Installation
---
Follow the [installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) as mentioned in the LightGBM GitHub [repository](https://github.com/microsoft/LightGBM).
Where needed, replace the repository ``https://github.com/microsoft/LightGBM`` with this one.

Citation
---

```
```
