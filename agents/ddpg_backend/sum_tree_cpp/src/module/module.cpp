#include <pybind11/pybind11.h>
#include <stdio.h>
#include "sum_tree.h"
#include "node.h"


PYBIND11_MODULE (sum_tree_cpp, module)
{
    module.doc () = "Pybind11Module";

    pybind11::class_<sum_tree> (module, "SumTreeCpp")
        .def (pybind11::init<int> (), pybind11::arg ("buffer_size"))
        .def (pybind11::init<int, double> (), pybind11::arg ("buffer_size"), pybind11::arg ("initial_priorities"))
        .def ("init", &sum_tree::init_tree)
        .def ("total", &sum_tree::total)
        .def ("size", &sum_tree::size)
        .def ("size_leaves", &sum_tree::size_leaves)
        .def ("update", &sum_tree::update, "updates the priority of a tree from a leave node upwards", pybind11::arg ("idx"), pybind11::arg ("priority"))
        .def ("update_mult", &sum_tree::update_mult, "updates multiple priorities in a tree, c.f. update()", pybind11::arg ("idxs"), pybind11::arg ("priorities"))
        .def ("get", &sum_tree::get, "get the idx of a leave node given a random variable s", pybind11::arg("s"))
        .def ("get_leaf_priorities", &sum_tree::get_leaf_priorities, "returns the leaf_nodes' priorities from start to end", pybind11::arg ("start"), pybind11::arg ("end"))
        .def_readwrite ("capacity", &sum_tree::buffer_size)
        .def_readwrite ("size_tree", &sum_tree::n_nodes)
    ;
}

