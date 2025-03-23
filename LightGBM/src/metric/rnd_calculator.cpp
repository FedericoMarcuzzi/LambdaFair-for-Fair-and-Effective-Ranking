/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
*/

#include <LightGBM/metric.h>
#include <LightGBM/utils/log.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace LightGBM {

void RNDCalculator::sort_by(std::vector<int> &group, const double* score) {
    std::vector<data_size_t> sorted_idx(group.size());
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(), [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    std::vector<int> l_app = group;

    for (size_t i = 0; i < group.size(); ++i)
        group[i] = l_app[sorted_idx[i]];
}

double RNDCalculator::rD(const std::vector<data_size_t> &group, size_t K, int truncation_level) {
    truncation_level = (truncation_level <= 0) ? group.size() : truncation_level;
    size_t end = std::min(static_cast<size_t>(truncation_level), group.size());

    size_t n_prot = std::accumulate(group.begin(), group.end(), 0);
    if (n_prot == 0 || n_prot == group.size())
        return 0.0;

    size_t slot = end / K + (end % K != 0);
    std::vector<size_t> slice_list(slot, K);
    if (end % K != 0)
        slice_list[slot - 1] = end % K;

    double sum = 0, rD = 0, p = n_prot / static_cast<double>(group.size());
    size_t last_step = 0, curr_step = 0;
    for (size_t step : slice_list) {
        curr_step += step;
        sum = std::accumulate(group.begin() + last_step, group.begin() + curr_step, sum);
        last_step = curr_step;

        rD += (1.0 / std::log2(curr_step + 1)) * fabs(sum / curr_step - p);
    }

    if (rD < 0.00000001)
        return 0.0;

    return rD;
}

double RNDCalculator::max_rD(const std::vector<data_size_t> &group, size_t K, int truncation_level) {
    truncation_level = (truncation_level <= 0) ? group.size() : truncation_level;
    size_t end = std::min(static_cast<size_t>(truncation_level), group.size());

    if (end <= K)
        return 0.0;

    int n_prot = std::accumulate(group.begin(), group.end(), 0);
    std::vector<data_size_t> l_app(group.size());

    std::fill(l_app.begin(), l_app.begin() + n_prot, 1);
    double a = rD(l_app, K, truncation_level);

    std::reverse(l_app.begin(), l_app.end());
    double b = rD(l_app, K, truncation_level);

    return std::max(a, b);
}

double RNDCalculator::rND(std::vector<data_size_t> group, const double* score, size_t K, double Z, int truncation_level) {
    truncation_level = (truncation_level <= 0) ? group.size() : truncation_level;
    double max = (Z < 0) ? max_rD(group, K, truncation_level) : Z;
    if (max <= 0.0)
        return 0.0;

    RNDCalculator::sort_by(group, score);
    return rD(group, K, truncation_level) / max;
}

}  // namespace LightGBM