#include <tuple>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
typedef py::array_t<int32_t> PyArrInt;
typedef py::array_t<float> PyArrFloat;

using Adj = std::tuple<PyArrInt, PyArrInt, PyArrFloat>;
using SingleSample = std::tuple<PyArrInt, Adj>;

SingleSample NodeWiseOneLayer(PyArrInt prev_nodes, PyArrInt indptr, PyArrInt indices, 
                            PyArrFloat values, int32_t layer_size, PyArrFloat probability, bool biased, bool replace);
PyArrInt LayerWiseOneLayer(PyArrInt indices, int32_t layer_size, PyArrFloat probability, bool biased, bool replace);