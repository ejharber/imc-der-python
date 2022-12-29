#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <pybind11/eigen.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include "eigenIncludes.h"
#include "world.h"
#include "setInput.h"

namespace py = pybind11;

PYBIND11_MODULE(pywrap, m)
{
    m.doc() = "pybind11 example plugin";

     py::class_<world>(m, "world")
        .def(py::init<string, int>())
        .def("getCurrentTime", &world::getCurrentTime)
        .def("setupSimulation", &world::setRodStepper)
        .def("stepSimulation", &world::updateTimeStep)
        .def("getStatePos", &world::getStatePos, py::return_value_policy::reference_internal)
        .def("getStateVel", &world::getStateVel, py::return_value_policy::reference_internal)
        .def("setPointVel", &world::setPointVel)
        .def("setStatePos", &world::setStatePos)
        .def("setStateVel", &world::setStateVel);
}
