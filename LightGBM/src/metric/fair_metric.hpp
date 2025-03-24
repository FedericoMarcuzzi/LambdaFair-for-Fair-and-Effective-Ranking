/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_METRIC_FAIR_METRIC_HPP_
#define LIGHTGBM_METRIC_FAIR_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

namespace LightGBM {

class FairMetric : public Metric {
   public:
    explicit FairMetric(const Config& config) {
        // get eval position
        rnd_step = config.rnd_step;
        eval_at_ = config.eval_at_rND;
    }

    ~FairMetric() {
    }
    void Init(const Metadata& metadata, data_size_t num_data) override {
        for (auto k : eval_at_)
            name_.emplace_back(std::string("rND@") + std::to_string(k));
        
        group_labels_str = metadata.group_labels();
        group_labels = metadata.eval_group_labels();

        /*
        group_labels.resize(num_data);
        for (size_t i = 0; i < group_labels.size(); ++i) {
            group_labels[i] = group_labels_str[i] - '0';
        }
        */

        num_queries_ = metadata.num_queries();
        // get query boundaries
        query_boundaries_ = metadata.query_boundaries();
        if (query_boundaries_ == nullptr)
            Log::Fatal("The rND metric requires query information");

        max_rnds_.resize(num_queries_);
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_queries_; ++i) {
            max_rnds_[i].resize(eval_at_.size(), 0.0f);
            std::vector<data_size_t> query_items_group(group_labels.begin() + query_boundaries_[i], group_labels.begin() + query_boundaries_[i + 1]);
            for (size_t j = 0; j < eval_at_.size(); ++j) {
                max_rnds_[i][j] = RNDCalculator::max_rD(query_items_group, rnd_step, eval_at_[j]);
            }
        }

        // get query weights
        query_weights_ = metadata.query_weights();
        if (query_weights_ == nullptr) {
            sum_query_weights_ = static_cast<double>(num_queries_);
        } else {
            sum_query_weights_ = 0.0f;
            for (data_size_t i = 0; i < num_queries_; ++i)
                sum_query_weights_ += query_weights_[i];
        }
    }

    const std::vector<std::string>& GetName() const override {
        return name_;
    }

    double factor_to_bigger_better() const override {
        return -1.0f;
    }

    std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
        int num_threads = OMP_NUM_THREADS();
        // some buffers for multi-threading sum up
        std::vector<std::vector<double>> result_buffer_;
        for (int i = 0; i < num_threads; ++i) {
            result_buffer_.emplace_back(eval_at_.size(), 0.0f);
        }
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_queries_; ++i) {
            const int tid = omp_get_thread_num();
            const double* query_score = score + query_boundaries_[i];
            std::vector<data_size_t> query_items_group(group_labels.begin() + query_boundaries_[i], group_labels.begin() + query_boundaries_[i + 1]);            
            for (size_t j = 0; j < eval_at_.size(); ++j) {
                result_buffer_[tid][j] += RNDCalculator::rND(query_items_group, query_score, rnd_step, max_rnds_[i][j], eval_at_[j]);
            }
        }

        // Get final average rND
        std::vector<double> result(eval_at_.size(), 0.0f);
        for (size_t j = 0; j < result.size(); ++j) {
            for (int i = 0; i < num_threads; ++i) {
                result[j] += result_buffer_[i][j];
            }
            result[j] /= sum_query_weights_;
        }
        return result;
    }

   private:
    int rnd_step;
    std::string group_labels_str;

    std::vector<int> group_labels;

    /*! \brief Name of test set */
    std::vector<std::string> name_;
    /*! \brief Query boundaries information */
    const data_size_t* query_boundaries_;
    /*! \brief Number of queries */
    data_size_t num_queries_;
    /*! \brief Weights of queries */
    const label_t* query_weights_;
    /*! \brief Sum weights of queries */
    double sum_query_weights_;
    /*! \brief Evaluate position of NDCG */
    std::vector<data_size_t> eval_at_;
    /*! \brief Cache the inverse max dcg for all queries */
    std::vector<std::vector<double>> max_rnds_;
};

}  // namespace LightGBM

#endif  // LightGBM_METRIC_FAIR_METRIC_HPP_
