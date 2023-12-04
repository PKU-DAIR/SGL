#include <random>

#include "sampling_ops.hpp"

std::mt19937 gen;

SingleSample NodeWiseOneLayer(PyArrInt prev_nodes, PyArrInt indptr, PyArrInt indices, PyArrFloat values, int32_t layer_size, PyArrFloat probability, bool biased, bool replace) {
    py::buffer_info buf_prev_nodes = prev_nodes.request();
    py::buffer_info buf_indptr = indptr.request();
    py::buffer_info buf_indices = indices.request();
    py::buffer_info buf_values = values.request();
    py::buffer_info buf_probability = probability.request();

    int32_t* ptr_prev_nodes = static_cast<int32_t *> (buf_prev_nodes.ptr);
    int32_t* ptr_indptr = static_cast<int32_t *> (buf_indptr.ptr);
    int32_t* ptr_indices = static_cast<int32_t *> (buf_indices.ptr);
    float* ptr_values = static_cast<float *> (buf_values.ptr);
    float* ptr_probability = static_cast<float *> (buf_probability.ptr);

    std::vector<std::vector<std::tuple<int32_t, float>>> cols; // col, v
    std::vector<int32_t> n_ids;
    std::unordered_map<int32_t, int32_t> n_id_map;
    
    auto out_indptr = PyArrInt(prev_nodes.size() + 1);
    py::buffer_info buf_out_indptr = out_indptr.request();
    int32_t* ptr_out_indptr = static_cast<int32_t *> (buf_out_indptr.ptr);
    ptr_out_indptr[0] = 0;

    int32_t n, c, e, start_, end_, num_neighbors;
    float v;

    for (int32_t i = 0; i < prev_nodes.size(); i++) {
        n = ptr_prev_nodes[i];
        cols.push_back(std::vector<std::tuple<int32_t, float>>());
        n_id_map[n] = i;
        n_ids.push_back(n);
    }

    if (layer_size < 0) {
        // No sampling
        for (int32_t i = 0; i < prev_nodes.size(); i++) {
            n = ptr_prev_nodes[i];
            start_ = ptr_indptr[n], end_ = ptr_indptr[n + 1];
            num_neighbors = end_ - start_;

            for (int32_t j = 0; j < num_neighbors; j++) {
                e = start_ + j;
                c = ptr_indices[e];
                v = ptr_values[e];

                if (n_id_map.count(c) == 0) {
                    n_id_map[c] = n_ids.size();
                    n_ids.push_back(c);
                }
                cols[i].push_back(std::make_tuple(n_id_map[c], v));
            }
            ptr_out_indptr[i + 1] = ptr_out_indptr[i] + cols[i].size();
        }
    } 
    else if (replace) {
        // Sample with replacement
        if (biased) {
            for (int32_t i = 0; i < prev_nodes.size(); i++) {
                n = ptr_prev_nodes[i];
                start_ = ptr_indptr[n], end_ = ptr_indptr[n + 1];
                num_neighbors = end_ - start_;
                
                if (num_neighbors > 0) {
                    std::vector<float> temp_probability(ptr_probability + start_, ptr_probability + end_);
                    for (int32_t j = 0; j < layer_size; j++) {
                        std::discrete_distribution<> d(temp_probability.begin(), temp_probability.end());
                        e = start_ + d(gen);
                        c = ptr_indices[e];
                        v = ptr_values[e];

                        if (n_id_map.count(c) == 0) {
                            n_id_map[c] = n_ids.size();
                            n_ids.push_back(c);
                        }
                        cols[i].push_back(std::make_tuple(n_id_map[c], v));
                    }
                }
                ptr_out_indptr[i + 1] = ptr_out_indptr[i] + cols[i].size();
            }
        }
        else {
            for (int32_t i = 0; i < prev_nodes.size(); i++) {
                n = ptr_prev_nodes[i];
                start_ = ptr_indptr[n], end_ = ptr_indptr[n + 1];
                num_neighbors = end_ - start_;
                
                if (num_neighbors > 0) {
                    for (int32_t j = 0; j < layer_size; j++) {
                        e = start_ + rand() % num_neighbors;
                        c = ptr_indices[e];
                        v = ptr_values[e];

                        if (n_id_map.count(c) == 0) {
                            n_id_map[c] = n_ids.size();
                            n_ids.push_back(c);
                        }
                        cols[i].push_back(std::make_tuple(n_id_map[c], v));
                    }
                }
                ptr_out_indptr[i + 1] = ptr_out_indptr[i] + cols[i].size();
            }
        }     
    }      
    else {
        // Sample without replacement
        if (biased) {
            for (int32_t i = 0; i < prev_nodes.size(); i++) {
                n = ptr_prev_nodes[i];
                start_ = ptr_indptr[n], end_ = ptr_indptr[n + 1];
                num_neighbors = end_ - start_;

                if (num_neighbors <= layer_size) {
                    for(int32_t j = 0; j < num_neighbors; j++) {
                        e = start_ + j;
                        c = ptr_indices[e];
                        v = ptr_values[e];

                        if (n_id_map.count(c) == 0) {
                            n_id_map[c] = n_ids.size();
                            n_ids.push_back(c);
                        }
                        cols[i].push_back(std::make_tuple(n_id_map[c], v));
                    }
                }
                else {
                    std::vector<float> temp_probability(ptr_probability + start_, ptr_probability + end_);
                    std::discrete_distribution<> d(temp_probability.begin(), temp_probability.end());
                    std::uniform_real_distribution<float> dist(0.0, 1.0);
                    std::vector<float> vals;
                    std::generate_n(std::back_inserter(vals), num_neighbors, [&dist]() { return dist(gen); });
                    std::transform(vals.begin(), vals.end(), temp_probability.begin(), vals.begin(), [&](auto r, auto prob) { return std::pow(r, 1. / prob); });
                    std::vector<std::pair<float, int32_t>> valIndices;
                    int32_t index = 0;
                    std::transform(vals.begin(), vals.end(), std::back_inserter(valIndices), [&index](auto v) { return std::pair<float, int32_t>(v, index++); });
                    std::sort(valIndices.begin(), valIndices.end(), [](auto x, auto y) { return x.first > y.first; });
                    std::vector<int32_t> candidates;
                    std::transform(valIndices.begin(), valIndices.end(), std::back_inserter(candidates), [](auto v) { return v.second; });
                    for(int32_t j = 0; j < layer_size; j++) {
                        e = start_ + candidates[j];
                        c = ptr_indices[e];
                        v = ptr_values[e];

                        if (n_id_map.count(c) == 0) {
                            n_id_map[c] = n_ids.size();
                            n_ids.push_back(c);
                        }
                        cols[i].push_back(std::make_tuple(n_id_map[c], v));
                    }
                }
                ptr_out_indptr[i + 1] = ptr_out_indptr[i] + cols[i].size();
            }
        }
        else {
            // via Robert Floyd algorithm
            for (int32_t i = 0; i < prev_nodes.size(); i++) {
                n = ptr_prev_nodes[i];
                start_ = ptr_indptr[n], end_ = ptr_indptr[n + 1];
                num_neighbors = end_ - start_;

                std::unordered_set<int32_t> perm;
                if (num_neighbors <= layer_size) {
                    for (int32_t j = 0; j < num_neighbors; j++) perm.insert(j);
                } else {
                    for (int32_t j = num_neighbors - layer_size; j < num_neighbors; j++) {
                        if (!perm.insert(rand() % j).second) perm.insert(j);
                    }
                }

                for(const int32_t &p: perm) {
                    e = start_ + p;
                    c = ptr_indices[e];
                    v = ptr_values[e];

                    if (n_id_map.count(c) == 0) {
                        n_id_map[c] = n_ids.size();
                        n_ids.push_back(c);
                    }
                    cols[i].push_back(std::make_tuple(n_id_map[c], v));
                }
                ptr_out_indptr[i + 1] = ptr_out_indptr[i] + cols[i].size();
            }
        }
    }

    int32_t E = ptr_out_indptr[prev_nodes.size()];
    auto out_indices = PyArrInt(E);
    py::buffer_info buf_out_indices = out_indices.request();
    int32_t* ptr_out_indices = static_cast<int32_t *> (buf_out_indices.ptr);
    auto out_values = PyArrFloat(E);
    py::buffer_info buf_out_values = out_values.request();
    float* ptr_out_values = static_cast<float *> (buf_out_values.ptr);

    int32_t i = 0;
    for (std::vector<std::tuple<int32_t, float>> &col_vec : cols) {
        std::sort(col_vec.begin(), col_vec.end(),
              [](const std::tuple<int64_t, float> &a,
                 const std::tuple<int64_t, float> &b) -> bool {
                return std::get<0>(a) < std::get<0>(b);
              });
        for (const std::tuple<int32_t, float> &value : col_vec) {
            ptr_out_indices[i] = std::get<0>(value);
            ptr_out_values[i] = std::get<1>(value);
            i += 1;
        }
    }

    PyArrInt out_n_ids(n_ids.size());
    py::buffer_info buf_out_n_ids = out_n_ids.request();
    int32_t *ptr_out_n_ids = static_cast<int32_t *>(buf_out_n_ids.ptr);
    std::copy(n_ids.begin(), n_ids.end(), ptr_out_n_ids);
    Adj out_adj = std::make_tuple(out_indptr, out_indices, out_values);
    return std::make_pair(out_n_ids, out_adj);
}

PyArrInt LayerWiseOneLayer(PyArrInt indices, int32_t layer_size, PyArrFloat probability, bool biased, bool replace) {
    py::buffer_info buf_indices = indices.request();
    py::buffer_info buf_probability = probability.request();

    int32_t* ptr_indices = static_cast<int32_t *> (buf_indices.ptr);
    float* ptr_probability = static_cast<float *> (buf_probability.ptr);

    std::vector<int32_t> neighbors(ptr_indices, ptr_indices + indices.size());
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    std::vector<int32_t> n_ids;
    int32_t e, c, num_neighbors = neighbors.size();

    if (layer_size < 0) {
        // No sampling
        n_ids.insert(n_ids.end(), neighbors.begin(), neighbors.end());
    } else if (replace) {
        // Sample with replacement
        n_ids.resize(layer_size);
        if (biased) {
            std::vector<float> selectedProbability(num_neighbors);
            std::transform(neighbors.begin(), neighbors.end(), selectedProbability.begin(),
                   [&ptr_probability](int index) { return ptr_probability[index]; }); 
            
            #pragma omp parallel for schedule(static)
            for (int32_t j = 0; j < layer_size; j++) {
                std::discrete_distribution<> d(selectedProbability.begin(), selectedProbability.end());
                e = d(gen);
                c = neighbors[e];
                n_ids[j] = c;
            }       
        } else {
            #pragma omp parallel for schedule(static)
            for (int32_t j = 0; j < layer_size; j++) {
                e = rand() % num_neighbors;
                c = neighbors[e];
                n_ids[j] = c;
            }
        }
    } else {
        // Sample without replacement
        if (num_neighbors <= layer_size) {
            n_ids.insert(n_ids.end(), neighbors.begin(), neighbors.end());
        } else if (biased) {
            std::vector<float> selectedProbability(num_neighbors);
            std::transform(neighbors.begin(), neighbors.end(), selectedProbability.begin(),
                   [&ptr_probability](int index) { return ptr_probability[index]; }); 
            std::discrete_distribution<> d(selectedProbability.begin(), selectedProbability.end());
            std::uniform_real_distribution<float> dist(0.0, 1.0);
            std::vector<float> vals;
            std::generate_n(std::back_inserter(vals), num_neighbors, [&dist]() { return dist(gen); });
            std::transform(vals.begin(), vals.end(), selectedProbability.begin(), vals.begin(), [&](auto r, auto prob) { return std::pow(r, 1. / prob); });
            std::vector<std::pair<float, int32_t>> valIndices;
            int32_t index = 0;
            std::transform(vals.begin(), vals.end(), std::back_inserter(valIndices), [&index](auto v) { return std::pair<float, int32_t>(v, index++); });
            std::sort(valIndices.begin(), valIndices.end(), [](auto x, auto y) { return x.first > y.first; });
            std::vector<int32_t> candidates;
            std::transform(valIndices.begin(), valIndices.end(), std::back_inserter(candidates), [](auto v) { return v.second; });

            n_ids.resize(layer_size);
            #pragma omp parallel for schedule(static)
            for (int32_t j = 0; j < layer_size; j++) {
                c = candidates[j];
                n_ids[j] = c;
            }
        } else {
            std::unordered_set<int32_t> perm;
            for (int32_t j = num_neighbors - layer_size; j < num_neighbors; j++) {
                if (!perm.insert(rand() % j).second) perm.insert(j);
            }          
            for (const int32_t &p: perm) {
                c = neighbors[p];
                n_ids.push_back(c);
            }
        }
    }

    PyArrInt out_n_ids(n_ids.size());
    py::buffer_info buf_out_n_ids = out_n_ids.request();
    int32_t *ptr_out_n_ids = static_cast<int32_t *>(buf_out_n_ids.ptr);
    std::copy(n_ids.begin(), n_ids.end(), ptr_out_n_ids);   
    return out_n_ids;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("NodeWiseOneLayer", &NodeWiseOneLayer);
    m.def("LayerWiseOneLayer", &LayerWiseOneLayer);
}