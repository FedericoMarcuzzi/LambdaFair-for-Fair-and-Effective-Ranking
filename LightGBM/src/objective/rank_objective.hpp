/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>
#include <bits/stdc++.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <typeinfo>
#include <vector>


using namespace std;

namespace LightGBM {

/*!
 * \brief Objective function for Ranking
 */
class RankingObjective : public ObjectiveFunction {
   public:
    explicit RankingObjective(const Config &config)
        : seed_(config.objective_seed) {
        learning_rate_ = config.learning_rate;
        position_bias_regularization_ = config.lambdarank_position_bias_regularization;
    }

    explicit RankingObjective(const std::vector<std::string> &) : seed_(0) {}

    ~RankingObjective() {}

    void Init(const Metadata &metadata, data_size_t num_data) override {
        num_data_ = num_data;
        // get label
        label_ = metadata.label();
        // get weights
        weights_ = metadata.weights();
        // get positions
        positions_ = metadata.positions();
        // get position ids
        position_ids_ = metadata.position_ids();
        // get number of different position ids
        num_position_ids_ = static_cast<data_size_t>(metadata.num_position_ids());
        // get boundries
        query_boundaries_ = metadata.query_boundaries();
        if (query_boundaries_ == nullptr) {
            Log::Fatal("Ranking tasks require query information");
        }

        // ***QUESTE SONO EFFETTIVAMENTE LE QUERY NEL TRAINING SET (RICAVATE GUARDANDO ULTIMA POSIZIONE DI OGNI RIGA)
        num_queries_ = metadata.num_queries();

        // initialize position bias vectors
        pos_biases_.resize(num_position_ids_, 0.0);
    }

    void GetGradients(const double *score, score_t *gradients,
                      score_t *hessians) const override {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(guided)
        for (data_size_t i = 0; i < num_queries_; ++i) {
            const data_size_t start = query_boundaries_[i];
            const data_size_t cnt = query_boundaries_[i + 1] - query_boundaries_[i];
            std::vector<double> score_adjusted;

            if (num_position_ids_ > 0)
                for (data_size_t j = 0; j < cnt; ++j)
                    score_adjusted.push_back(score[start + j] + pos_biases_[positions_[start + j]]);

            GetGradientsForOneQuery(i, cnt, label_ + start, num_position_ids_ > 0 ? score_adjusted.data() : score + start, gradients + start, hessians + start);

            if (weights_ != nullptr) {
                for (data_size_t j = 0; j < cnt; ++j) {
                    gradients[start + j] =
                        static_cast<score_t>(gradients[start + j] * weights_[start + j]);
                    hessians[start + j] =
                        static_cast<score_t>(hessians[start + j] * weights_[start + j]);
                }
            }
        }

        if (num_position_ids_ > 0)
            UpdatePositionBiasFactors(gradients, hessians);
    }

    virtual void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                         const label_t *label,
                                         const double *score, score_t *lambdas,
                                         score_t *hessians) const = 0;

    virtual void UpdatePositionBiasFactors(const score_t * /*lambdas*/, const score_t * /*hessians*/) const {}

    const char *GetName() const override = 0;

    std::string ToString() const override {
        std::stringstream str_buf;
        str_buf << GetName();
        return str_buf.str();
    }

    bool NeedAccuratePrediction() const override { return false; }

   protected:
    int seed_;
    data_size_t num_queries_;
    /*! \brief Number of data */
    data_size_t num_data_;
    /*! \brief Pointer of label */
    const label_t *label_;
    /*! \brief Pointer of weights */
    const label_t *weights_;
    /*! \brief Pointer of positions */
    const data_size_t *positions_;
    /*! \brief Pointer of position IDs */
    const std::string *position_ids_;
    /*! \brief Pointer of label */
    data_size_t num_position_ids_;
    /*! \brief Query boundaries */
    const data_size_t *query_boundaries_;
    /*! \brief Position bias factors */
    mutable std::vector<label_t> pos_biases_;
    /*! \brief Learning rate to update position bias factors */
    double learning_rate_;
    /*! \brief Position bias regularization */
    double position_bias_regularization_;
};

/*!
 * \brief Objective function for LambdaRank with NDCG
 */
class LambdarankNDCG : public RankingObjective {
   public:
    explicit LambdarankNDCG(const Config &config)
        : RankingObjective(config),
          sigmoid_(config.sigmoid),
          norm_(config.lambdarank_norm),
          truncation_level_(config.lambdarank_truncation_level) {
        label_gain_ = config.label_gain;

        std::map<std::string, size_t> strategy_map = {{"plain", 0}, {"ndcg", 1}, {"rnd", 2}, {"delta", 3}};
        strategy_ = strategy_map[config.lambda_fair];
        is_normal_train = strategy_ == 0;

        rnd_step = config.rnd_step;
        alpha = config.alpha_lambdafair;
        beta = 1.0 - alpha;
        group = config.group_labels;

        // initialize DCG calculator
        DCGCalculator::DefaultLabelGain(&label_gain_);
        DCGCalculator::Init(label_gain_);
        sigmoid_table_.clear();
        inverse_max_dcgs_.clear();
        if (sigmoid_ <= 0.0) {
            Log::Fatal("Sigmoid param %f should be greater than zero", sigmoid_);
        }
    }

    explicit LambdarankNDCG(const std::vector<std::string> &strs)
        : RankingObjective(strs) {}

    ~LambdarankNDCG() {}

    void Init(const Metadata &metadata, data_size_t num_data) override {
        RankingObjective::Init(metadata, num_data);
        DCGCalculator::CheckMetadata(metadata, num_queries_);
        DCGCalculator::CheckLabel(label_, num_data_);

        inverse_max_dcgs_.resize(num_queries_);
        max_rnds.resize(num_queries_);
        
        /* gets higher relevance label */
        max_label = 0;
        for (size_t i = 0; i < group.size(); ++i) {
            if (max_label < label_[i])
                max_label = label_[i];
        }

        prot_per_qry_lbl_blk.resize(num_queries_);
        unprot_per_qry_lbl_blk.resize(num_queries_);
        std::vector<data_size_t> qry_lens(num_queries_);

        /* gets higher query length and compute max rNDs */
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_queries_; ++i) {
            const std::vector<data_size_t> qry_group(group.begin() + query_boundaries_[i], group.begin() + query_boundaries_[i + 1]);
            max_rnds[i] = RNDCalculator::max_rD(qry_group, rnd_step, truncation_level_);
            qry_lens[i] = query_boundaries_[i + 1] - query_boundaries_[i];
        }
        data_size_t max_query_len = *std::max_element(qry_lens.begin(), qry_lens.end());

        rND_k_disc.resize(max_query_len / rnd_step);    // rND
        rND_log_disc.resize(max_query_len / rnd_step);  // rND

        if (!is_normal_train) {                                          // rND
            for (data_size_t i = 0; i < max_query_len / rnd_step; ++i) {        // rND
                rND_log_disc[i] = 1.0 / std::log2(((i + 1) * rnd_step) + 1);    // rND
                rND_k_disc[i] = 1.0 / ((i + 1) * rnd_step);                     // rND
            }                                                            // rND
        }                                                                // rND

        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t query_id = 0; query_id < num_queries_; ++query_id) {
            const int cnt = query_boundaries_[query_id + 1] - query_boundaries_[query_id];
            const int n_blk = std::ceil((cnt * 1.0) / rnd_step);

            prot_per_qry_lbl_blk[query_id].resize(max_label + 1, std::vector<int>(n_blk));
            unprot_per_qry_lbl_blk[query_id].resize(max_label + 1, std::vector<int>(n_blk));

            std::vector<data_size_t> qry_group(group.begin() + query_boundaries_[query_id], group.begin() + query_boundaries_[query_id + 1]);
            int n_prot_qry = std::accumulate(qry_group.begin(), qry_group.end(), 0);
            double p_prot = (n_prot_qry * 1.0) / cnt;

            std::vector<int> prot_per_blk(n_blk, 0);
            std::vector<int> unprot_per_blk(n_blk, 0);

            // looks for the best distribution of protected and unprotected items according to rND
            for (data_size_t b = 0, n_prot_placed = 0; b < n_blk; ++b) {
                float n_items  = std::min(cnt, rnd_step * (b + 1));
                double min_p_diff = fabs(n_prot_placed / n_items - p_prot);
                while(n_prot_placed < n_prot_qry) {
                    double p_diff = fabs((n_prot_placed + 1) / n_items - p_prot);
                    if (p_diff >= min_p_diff)
                        break;

                    min_p_diff = p_diff;
                    n_prot_placed++;
                    prot_per_blk[b]++;
                }

                if ((n_blk - 1) == b) {
                    if (cnt % rnd_step == 0)
                        unprot_per_blk[b] = rnd_step - prot_per_blk[b];
                    else
                        unprot_per_blk[b] = (cnt % rnd_step) - prot_per_blk[b];
                } else
                    unprot_per_blk[b] = rnd_step - prot_per_blk[b];
            }

            const label_t *qry_label = label_ + query_boundaries_[query_id];
            std::vector<data_size_t> sorted_idx(cnt);
            std::iota(sorted_idx.begin(), sorted_idx.end(), 0);            

            // optimize wrt rND metric
            if (strategy_ == 2) {
                std::stable_sort(
                sorted_idx.begin(), sorted_idx.end(), 
                [qry_label](data_size_t a, data_size_t b) { return qry_label[a] > qry_label[b]; });

                for (data_size_t j : sorted_idx) {
                    label_t l = qry_label[j];
                    data_size_t b = 0;
                    while ((qry_group[j] ? prot_per_blk[b] : unprot_per_blk[b]) == 0) { b++; }

                    ((qry_group[j]) ? ++prot_per_qry_lbl_blk[query_id][l][b] : ++unprot_per_qry_lbl_blk[query_id][l][b]);
                    ((qry_group[j]) ? --prot_per_blk[b] : --unprot_per_blk[b]);
                }
            }

            if (strategy_ == 1) { // optimize wrt NDCG metric
                std::stable_sort(
                sorted_idx.begin(), sorted_idx.end(), 
                [qry_label, qry_group](data_size_t a, data_size_t b) { return (qry_label[a] > qry_label[b]) ? true : ((qry_label[a] == qry_label[b]) && (qry_group[a] > qry_group[b])); });

                for (data_size_t b = 1; b < n_blk; ++b) {
                    prot_per_blk[b] = prot_per_blk[b] + prot_per_blk[b - 1];
                    unprot_per_blk[b] = unprot_per_blk[b] + unprot_per_blk[b - 1];
                }

                int prot_placed = 0, unprot_placed = 0;
                for (data_size_t i = 0, last_pos = 1; i < cnt; ++i) {
                    data_size_t doc_id = sorted_idx[i], b = i / rnd_step;
                    label_t l = qry_label[doc_id];
                    const int& g = qry_group[doc_id];

                    int& per_blk = (g) ? prot_per_blk[b] : unprot_per_blk[b];
                    int& first_placed = (g) ? prot_placed : unprot_placed;
                    int& second_placed = (g) ? unprot_placed : prot_placed;
                    int& first_per_blk = (g) ? prot_per_qry_lbl_blk[query_id][l][b] : unprot_per_qry_lbl_blk[query_id][l][b];
                    int& second_per_blk = (g) ? unprot_per_qry_lbl_blk[query_id][l][b] : prot_per_qry_lbl_blk[query_id][l][b];

                    if(per_blk - first_placed > 0) {
                        first_per_blk++;
                        first_placed++;
                    } else {
                        last_pos = std::max(i + 1, last_pos);
                        bool found = false;
                        while((last_pos < cnt) && (qry_label[sorted_idx[last_pos]] == l) && !found) {
                            if ((qry_label[sorted_idx[last_pos]] == l) && (qry_group[doc_id] != qry_group[sorted_idx[last_pos]])) {
                                std::swap(qry_group[doc_id], qry_group[sorted_idx[last_pos]]);     
                                found = true;
                            }
                            ++last_pos;
                        }

                        if(found) {
                            second_per_blk++;
                            second_placed++;
                        } else {
                            first_per_blk++;
                            first_placed++;
                        }
                    }
                }
            }
        }

        for (data_size_t i = 0; i < num_queries_; ++i) {
            inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(
                truncation_level_, label_ + query_boundaries_[i],
                query_boundaries_[i + 1] - query_boundaries_[i]);

            if (inverse_max_dcgs_[i] > 0.0) {
                inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
            }
        }
        // construct Sigmoid table to speed up Sigmoid transform
        ConstructSigmoidTable();
    }

    inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                        const label_t *label, const double *score,
                                        score_t *lambdas,
                                        score_t *hessians) const override {
        double p_prot = 0;
        int cum_n_prot = 0;                                         // rND
        // get max inverse DCG on current query
        const double inverse_max_dcg = inverse_max_dcgs_[query_id];
        // get max rND on current query
        const double Z = max_rnds[query_id];                        // rND
        const int n_blk = std::ceil((cnt * 1.0) / rnd_step);        // rND

        std::vector<score_t> lambdas_rnd(cnt);                      // rND
        std::vector<score_t> hessians_rnd(cnt);                     // rND

        std::vector<int> n_rnk_prot_per_blk(cnt / rnd_step);        // rND
        std::vector<int> n_rnk_prot_per_blk_minus(cnt / rnd_step);  // rND
        std::vector<int> n_rnk_prot_per_blk_plus(cnt / rnd_step);   // rND
        std::vector<int> rND_labels(cnt);                           // rND

        std::vector<data_size_t> qry_group(group.begin() + query_boundaries_[query_id], group.begin() + query_boundaries_[query_id + 1]); 

        std::vector<int> prot_pos(max_label + 1, 0);
        std::vector<int> no_prot_pos(max_label + 1, 0);

        std::vector<std::vector<int>> prot_per_lbl_blk = prot_per_qry_lbl_blk[query_id];
        std::vector<std::vector<int>> unprot_per_lbl_blk = unprot_per_qry_lbl_blk[query_id];

        // get sorted indices for scores
        std::vector<data_size_t> sorted_idx(cnt);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::stable_sort(
            sorted_idx.begin(), sorted_idx.end(),
            [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });

        std::vector<data_size_t> sorted_idx_rnd(cnt);
        std::iota(sorted_idx_rnd.begin(), sorted_idx_rnd.end(), 0);
        std::stable_sort(
            sorted_idx_rnd.begin(), sorted_idx_rnd.end(), 
            [label, score](data_size_t a, data_size_t b) { return (label[a] > label[b]) ? true : ((label[a] == label[b]) && (score[a] > score[b])); });


        // initialize with zero
        for (data_size_t i = 0, j = 0; i < cnt; ++i) {
            lambdas[i] = 0.0f;
            hessians[i] = 0.0f;

            cum_n_prot += qry_group[sorted_idx[i]];            // rND
            if ((i + 1) % rnd_step == 0) {                     // rND
                n_rnk_prot_per_blk[j] = cum_n_prot;            // rND
                n_rnk_prot_per_blk_minus[j] = cum_n_prot - 1;  // rND
                n_rnk_prot_per_blk_plus[j] = cum_n_prot + 1;   // rND
                ++j;
            }

            if (strategy_ == 1 or strategy_ == 2) {
                label_t doc_l = label[sorted_idx_rnd[i]];
                bool test_prot = qry_group[sorted_idx_rnd[i]] == 1;
                std::vector<int> &v = (test_prot) ? prot_per_lbl_blk[doc_l] : unprot_per_lbl_blk[doc_l];
                int &p = (test_prot) ? prot_pos[doc_l] : no_prot_pos[doc_l];
                while (v[p] == 0) { p++; }

                rND_labels[sorted_idx_rnd[i]] = n_blk - p;
                v[p]--;
            }
        }

        p_prot = cum_n_prot / static_cast<double>(cnt);  // rND

        // get best and worst score
        const double best_score = score[sorted_idx[0]];
        data_size_t worst_idx = cnt - 1;
        if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
            worst_idx -= 1;
        }
        const double worst_score = score[sorted_idx[worst_idx]];
        double sum_lambdas = 0.0;
        double sum_lambdas_rND = 0.0;

        bool test_same_rel_lab = false;   // rND
        bool test_same_prot_grp = false;  // rND

        for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
            if (score[sorted_idx[i]] == kMinScore) {
                continue;
            }

            for (data_size_t j = i + 1; j < cnt; ++j) {
                if (score[sorted_idx[j]] == kMinScore) {
                    continue;
                }
                // skip pairs with the same labels
                test_same_rel_lab = label[sorted_idx[i]] == label[sorted_idx[j]];  // rND
                // skip pairs with the same protected value
                test_same_prot_grp = qry_group[sorted_idx[i]] == qry_group[sorted_idx[j]];  // rND

                if (is_normal_train && test_same_rel_lab) {
                    continue;
                }
            
                if (!is_normal_train && test_same_rel_lab && test_same_prot_grp) {
                    continue;
                }  // rND

                // rND : delta_pair_rND --> START
                if (!is_normal_train && !test_same_prot_grp && Z > 0.0 && beta > 0.0) {
                    // start accmulate lambdas by pairs that contain at least one document above truncation level

                    // computes delta_pair_rND -> start
                    double cumsum_ij = 0.0, cumsum_ji = 0.0, delta_pair_rND = 0.0;  // rND
                    bool is_j_prot = qry_group[sorted_idx[i]] < qry_group[sorted_idx[j]];

                    int stop_pos = std::min(j, truncation_level_);
                    for (data_size_t k = i / rnd_step; k < stop_pos / rnd_step; ++k) {
                        cumsum_ij = fabs(n_rnk_prot_per_blk[k] * rND_k_disc[k] - p_prot);
                        cumsum_ji = fabs((is_j_prot ? n_rnk_prot_per_blk_plus[k] : n_rnk_prot_per_blk_minus[k]) * rND_k_disc[k] - p_prot);
                        delta_pair_rND += rND_log_disc[k] * (cumsum_ij - cumsum_ji);
                    }

                    if (truncation_level_ % rnd_step != 0 && truncation_level_ <= j) {
                        cumsum_ij = (truncation_level_ < rnd_step) ? 0 : n_rnk_prot_per_blk[truncation_level_ / rnd_step - 1];
                        cumsum_ji = is_j_prot ? cumsum_ij + 1 : cumsum_ij - 1;

                        stop_pos = truncation_level_ - (truncation_level_ % rnd_step);
                        for (; stop_pos < truncation_level_; ++stop_pos) {
                            cumsum_ij += qry_group[sorted_idx[stop_pos]];
                            cumsum_ji += qry_group[sorted_idx[stop_pos]];
                        }

                        cumsum_ij = fabs(cumsum_ij / truncation_level_ - p_prot);
                        cumsum_ji =  fabs(cumsum_ji / truncation_level_ - p_prot);
                        delta_pair_rND += (1.0 / std::log2(truncation_level_ + 1)) * (cumsum_ij - cumsum_ji);
                    }
                    // end

                    bool is_better_i_higher = false;
                    if (strategy_ == 0)
                        is_better_i_higher = label[sorted_idx[i]] > label[sorted_idx[j]];
                    if (strategy_ == 1 or strategy_ == 2)
                        is_better_i_higher = rND_labels[sorted_idx[i]] > rND_labels[sorted_idx[j]];
                    if (strategy_ == 3)
                        is_better_i_higher = delta_pair_rND < 0;

                    data_size_t high_rank_rND, low_rank_rND;
                    if (is_better_i_higher) {
                        high_rank_rND = i;
                        low_rank_rND = j;
                    } else {
                        high_rank_rND = j;
                        low_rank_rND = i;
                    }

                    delta_pair_rND = fabs(delta_pair_rND) / Z;

                    const data_size_t high_rND = sorted_idx[high_rank_rND];
                    const data_size_t low_rND = sorted_idx[low_rank_rND];

                    const double delta_score_rND = score[high_rND] - score[low_rND];
                    if (norm_ && best_score != worst_score) {
                        delta_pair_rND /= (0.01f + fabs(delta_score_rND));  // rND
                    }

                    double p_lambda_rND = GetSigmoid(delta_score_rND);            // rND
                    double p_hessian_rND = p_lambda_rND * (1.0f - p_lambda_rND);  // rND
                    p_lambda_rND *= -sigmoid_ * delta_pair_rND * beta;            // rND
                    p_hessian_rND *= sigmoid_ * sigmoid_ * delta_pair_rND * beta;

                    lambdas_rnd[low_rND] -= static_cast<score_t>(p_lambda_rND);
                    hessians_rnd[low_rND] += static_cast<score_t>(p_hessian_rND);
                    lambdas_rnd[high_rND] += static_cast<score_t>(p_lambda_rND);
                    hessians_rnd[high_rND] += static_cast<score_t>(p_hessian_rND);

                    sum_lambdas_rND -= 2 * p_lambda_rND;
                }  // rND : delta_pair_rND <-- END

                if (!test_same_rel_lab && alpha > 0.0) {
                    data_size_t high_rank, low_rank;
                    if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
                        high_rank = i;
                        low_rank = j;
                    } else {
                        high_rank = j;
                        low_rank = i;
                    }

                    const data_size_t high = sorted_idx[high_rank];
                    const int high_label = static_cast<int>(label[high]);
                    const double high_score = score[high];
                    const double high_label_gain = label_gain_[high_label];
                    const double high_discount = DCGCalculator::GetDiscount(high_rank);
                    const data_size_t low = sorted_idx[low_rank];
                    const int low_label = static_cast<int>(label[low]);
                    const double low_score = score[low];
                    const double low_label_gain = label_gain_[low_label];
                    const double low_discount = DCGCalculator::GetDiscount(low_rank);

                    const double delta_score = high_score - low_score;

                    // get dcg gap
                    const double dcg_gap = high_label_gain - low_label_gain;
                    // get discount of this pair
                    const double paired_discount = (j < truncation_level_) ? fabs(high_discount - low_discount) : DCGCalculator::GetDiscount(i);
                    // get delta NDCG
                    double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
                    // regular the delta_pair_NDCG by score distance
                    if (norm_ && best_score != worst_score) {
                        delta_pair_NDCG /= (0.01f + fabs(delta_score));
                    }
                    // calculate lambda for this pair
                    double p_lambda_NDCG = GetSigmoid(delta_score);
                    double p_hessian_NDCG = p_lambda_NDCG * (1.0f - p_lambda_NDCG);

                    // update
                    p_lambda_NDCG *= -sigmoid_ * delta_pair_NDCG * alpha;
                    p_hessian_NDCG *= sigmoid_ * sigmoid_ * delta_pair_NDCG * alpha;

                    lambdas[low] -= static_cast<score_t>(p_lambda_NDCG);
                    hessians[low] += static_cast<score_t>(p_hessian_NDCG);
                    lambdas[high] += static_cast<score_t>(p_lambda_NDCG);
                    hessians[high] += static_cast<score_t>(p_hessian_NDCG);

                    // lambda is negative, so use minus to accumulate
                    sum_lambdas -= 2 * p_lambda_NDCG;
                }
            }
        }

        if (norm_ && sum_lambdas > 0) {
            double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;

            for (data_size_t i = 0; i < cnt; ++i) {
                lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
                hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
            }
        }

        if (norm_ && sum_lambdas_rND > 0) {
            double norm_factor = std::log2(1 + sum_lambdas_rND) / sum_lambdas_rND;

            for (data_size_t i = 0; i < cnt; ++i) {
                lambdas_rnd[i] = static_cast<score_t>(lambdas_rnd[i] * norm_factor);
                hessians_rnd[i] = static_cast<score_t>(hessians_rnd[i] * norm_factor);
            }
        }

        for (data_size_t i = 0; i < cnt; ++i) {
            lambdas[i] += lambdas_rnd[i];
            hessians[i] += hessians_rnd[i];
        }
    }

    inline double GetSigmoid(double score) const {
        if (score <= min_sigmoid_input_) {
            // too small, use lower bound
            return sigmoid_table_[0];
        } else if (score >= max_sigmoid_input_) {
            // too large, use upper bound
            return sigmoid_table_[_sigmoid_bins - 1];
        } else {
            return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) *
                                                      sigmoid_table_idx_factor_)];
        }
    }

    void ConstructSigmoidTable() {
        // get boundary
        min_sigmoid_input_ = min_sigmoid_input_ / sigmoid_ / 2;
        max_sigmoid_input_ = -min_sigmoid_input_;
        sigmoid_table_.resize(_sigmoid_bins);
        // get score to bin factor
        sigmoid_table_idx_factor_ =
            _sigmoid_bins / (max_sigmoid_input_ - min_sigmoid_input_);
        // cache
        for (size_t i = 0; i < _sigmoid_bins; ++i) {
            const double score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_;
            sigmoid_table_[i] = 1.0f / (1.0f + std::exp(score * sigmoid_));
        }
    }

    void UpdatePositionBiasFactors(const score_t *lambdas, const score_t *hessians) const override {
        /// get number of threads
        int num_threads = OMP_NUM_THREADS();
        // create per-thread buffers for first and second derivatives of utility w.r.t. position bias factors
        std::vector<double> bias_first_derivatives(num_position_ids_ * num_threads, 0.0);
        std::vector<double> bias_second_derivatives(num_position_ids_ * num_threads, 0.0);
        std::vector<int> instance_counts(num_position_ids_ * num_threads, 0);
        #pragma omp parallel for schedule(guided) num_threads(num_threads)
        for (data_size_t i = 0; i < num_data_; i++) {
            // get thread ID
            const int tid = omp_get_thread_num();
            size_t offset = static_cast<size_t>(positions_[i] + tid * num_position_ids_);
            // accumulate first derivatives of utility w.r.t. position bias factors, for each position
            bias_first_derivatives[offset] -= lambdas[i];
            // accumulate second derivatives of utility w.r.t. position bias factors, for each position
            bias_second_derivatives[offset] -= hessians[i];
            instance_counts[offset]++;
        }

        #pragma omp parallel for schedule(guided) num_threads(num_threads)
        for (data_size_t i = 0; i < num_position_ids_; i++) {
            double bias_first_derivative = 0.0;
            double bias_second_derivative = 0.0;
            int instance_count = 0;
            // aggregate derivatives from per-thread buffers
            for (int tid = 0; tid < num_threads; tid++) {
                size_t offset = static_cast<size_t>(i + tid * num_position_ids_);
                bias_first_derivative += bias_first_derivatives[offset];
                bias_second_derivative += bias_second_derivatives[offset];
                instance_count += instance_counts[offset];
            }
            // L2 regularization on position bias factors
            bias_first_derivative -= pos_biases_[i] * position_bias_regularization_ * instance_count;
            bias_second_derivative -= position_bias_regularization_ * instance_count;
            // do Newton-Raphson step to update position bias factors
            pos_biases_[i] += learning_rate_ * bias_first_derivative / (std::abs(bias_second_derivative) + 0.001);
        }
        LogDebugPositionBiasFactors();
    }

    const char *GetName() const override { return "lambdarank"; }

   protected:
    void LogDebugPositionBiasFactors() const {
        std::stringstream message_stream;
        message_stream << std::setw(15) << "position"
                       << std::setw(15) << "bias_factor"
                       << std::endl;
        Log::Debug(message_stream.str().c_str());
        message_stream.str("");
        for (int i = 0; i < num_position_ids_; ++i) {
            message_stream << std::setw(15) << position_ids_[i]
                           << std::setw(15) << pos_biases_[i];
            Log::Debug(message_stream.str().c_str());
            message_stream.str("");
        }
    }

    // rND --> START
    /*! \brief LambdaFair strategy */
    size_t strategy_ = 0;
    int max_label = 0;
    int rnd_step = 5;
    double alpha = 1.0;
    double beta = 0.0;
    bool is_normal_train = 1;
    std::vector<data_size_t> group;
    std::vector<double> max_rnds;
    std::vector<double> rND_log_disc;
    std::vector<double> rND_k_disc;
    mutable std::vector<std::vector<std::vector<int>>> prot_per_qry_lbl_blk;
    mutable std::vector<std::vector<std::vector<int>>> unprot_per_qry_lbl_blk;
    // END

    /*! \brief Sigmoid param */
    double sigmoid_;
    /*! \brief Normalize the lambdas or not */
    bool norm_;
    /*! \brief Truncation position for max DCG */
    int truncation_level_;
    /*! \brief Cache inverse max DCG, speed up calculation */
    std::vector<double> inverse_max_dcgs_;
    /*! \brief Cache result for sigmoid transform to speed up */
    std::vector<double> sigmoid_table_;
    /*! \brief Gains for labels */
    std::vector<double> label_gain_;
    /*! \brief Number of bins in simoid table */
    size_t _sigmoid_bins = 1024 * 1024;
    /*! \brief Minimal input of sigmoid table */
    double min_sigmoid_input_ = -50;
    /*! \brief Maximal input of Sigmoid table */
    double max_sigmoid_input_ = 50;
    /*! \brief Factor that covert score to bin in Sigmoid table */
    double sigmoid_table_idx_factor_;
};

/*!
 * \brief Implementation of the learning-to-rank objective function, XE_NDCG
 * [arxiv.org/abs/1911.09798].
 */
class RankXENDCG : public RankingObjective {
   public:
    explicit RankXENDCG(const Config &config) : RankingObjective(config) {}

    explicit RankXENDCG(const std::vector<std::string> &strs)
        : RankingObjective(strs) {}

    ~RankXENDCG() {}

    void Init(const Metadata &metadata, data_size_t num_data) override {
        RankingObjective::Init(metadata, num_data);
        for (data_size_t i = 0; i < num_queries_; ++i) {
            rands_.emplace_back(seed_ + i);
        }
    }

    inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                        const label_t *label, const double *score,
                                        score_t *lambdas,
                                        score_t *hessians) const override {
        // Skip groups with too few items.
        if (cnt <= 1) {
            for (data_size_t i = 0; i < cnt; ++i) {
                lambdas[i] = 0.0f;
                hessians[i] = 0.0f;
            }
            return;
        }

        // Turn scores into a probability distribution using Softmax.
        std::vector<double> rho(cnt, 0.0);
        Common::Softmax(score, rho.data(), cnt);

        // An auxiliary buffer of parameters used to form the ground-truth
        // distribution and compute the loss.
        std::vector<double> params(cnt);

        double inv_denominator = 0;
        for (data_size_t i = 0; i < cnt; ++i) {
            params[i] = Phi(label[i], rands_[query_id].NextFloat());
            inv_denominator += params[i];
        }
        // sum_labels will always be positive number
        inv_denominator = 1. / std::max<double>(kEpsilon, inv_denominator);

        // Approximate gradients and inverse Hessian.
        // First order terms.
        double sum_l1 = 0.0;
        for (data_size_t i = 0; i < cnt; ++i) {
            double term = -params[i] * inv_denominator + rho[i];
            lambdas[i] = static_cast<score_t>(term);
            // Params will now store terms needed to compute second-order terms.
            params[i] = term / (1. - rho[i]);
            sum_l1 += params[i];
        }
        // Second order terms.
        double sum_l2 = 0.0;
        for (data_size_t i = 0; i < cnt; ++i) {
            double term = rho[i] * (sum_l1 - params[i]);
            lambdas[i] += static_cast<score_t>(term);
            // Params will now store terms needed to compute third-order terms.
            params[i] = term / (1. - rho[i]);
            sum_l2 += params[i];
        }
        for (data_size_t i = 0; i < cnt; ++i) {
            lambdas[i] += static_cast<score_t>(rho[i] * (sum_l2 - params[i]));
            hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
        }
    }

    double Phi(const label_t l, double g) const {
        return Common::Pow(2, static_cast<int>(l)) - g;
    }

    const char *GetName() const override { return "rank_xendcg"; }

   protected:
    mutable std::vector<Random> rands_;
};

}  // namespace LightGBM
#endif  // LightGBM_OBJECTIVE_RANK_OBJECTIVE_HPP[_