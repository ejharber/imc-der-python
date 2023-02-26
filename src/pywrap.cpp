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

        .def("getStateD1", &world::getStateD1, py::return_value_policy::reference_internal)
        .def("getStateD2", &world::getStateD2, py::return_value_policy::reference_internal) // potentially unneccesarry
        .def("getStateTangent", &world::getStateTangent, py::return_value_policy::reference_internal)
        .def("getStateRefTwist", &world::getStateRefTwist, py::return_value_policy::reference_internal)

        .def("setStatePos", &world::setStatePos)
        .def("setStateVel", &world::setStateVel)

        .def("setStateD1", &world::setStateD1, py::return_value_policy::reference_internal)
        .def("setStateD2", &world::setStateD2, py::return_value_policy::reference_internal) // potentially unnecessary
        .def("setStateTangent", &world::setStateTangent, py::return_value_policy::reference_internal)
        .def("setStateRefTwist", &world::setStateRefTwist, py::return_value_policy::reference_internal)

        .def("setPointVel", &world::setPointVel);
}
