#include "eigensolver.hpp"
#include "HamiltonianBase.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(eigen_wrapper, m) {
    m.doc() = "Interface to call Eigen C++ functions from python";

    m.def("eigen", &eigen, "Calculate eigenvaues of the Hamiltonian", py::return_value_policy::copy);

    py::class_<HamiltonianBase>(m, "Hamiltonian")
        .def(py::init<std::vector<int>, std::vector<double>, std::vector<double>, double>())
        .def("set_potential", &HamiltonianBase::set_potential )
        .def("as_operator", &HamiltonianBase::as_operator);
}
