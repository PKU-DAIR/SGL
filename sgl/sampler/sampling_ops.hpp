#include <tuple>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
typedef py::array_t<int32_t> PyArrInt;
typedef py::array_t<float> PyArrFloat;

using Adj = std::tuple<PyArrInt, PyArrInt, PyArrFloat>;
using Adjs = std::vector<Adj>;
using SingleSample = std::tuple<PyArrInt, Adj>;
using BatchSamples = std::tuple<PyArrInt, PyArrInt, Adjs>;

SingleSample NodeWiseOneLayer(PyArrInt prev_nodes, PyArrInt indptr, PyArrInt indices, 
                            PyArrFloat values, int32_t layer_size, PyArrFloat probability, bool biased, bool replace);
// BatchSamples NodeWiseMultiLayers(PyArrInt batch_inds, PyArrInt indptr, PyArrInt indices, 
//                                 PyArrFloat values, PyArrInt layer_sizes, PyArrFloat probability, 
//                                 bool biased, bool replace);