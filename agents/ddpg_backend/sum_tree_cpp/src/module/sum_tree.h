#ifndef SUM_TREE_H
#define SUM_TREE_H

#include "node.h"
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

class sum_tree
{
public:
    int buffer_size;
    int n_nodes;
    node* root;
    std::vector<node*> tree;
    std::vector<node*> leaf_nodes;

    sum_tree(int buffer_size, double init_priorities=0.0);
    ~sum_tree();

    /// initialize the tree
    void init_tree(double init_priority);
    /// updates the priority of a tree from a leave node upwards
    void update(int leaf_idx, double priority);
    /// updates multiple priorities in a tree, c.f. update()
    void update_mult(pybind11::array_t<int> leaf_idxs, pybind11::array_t<double> priorities);
    /// get the idx and the priority of a leave node given a random variable s
    pybind11::tuple get(double s);
    /// returns the leaf_nodes' priorities from start to end
    pybind11::list get_leaf_priorities(int start, int end);
    /// returns the sum of all priorities in the leaf nodes
    double total();
    /// returns the total number of nodes in a tree
    int size();
    /// returns the total number of leaf nodes
    int size_leaves();

private:
    /// sets a given change in priority in a node and propagates the change up to the root node
    void propagate(node* node_ptr, double change);
    /// returns a leave node given a random variable s
    node* retrieve(node* node_ptr, double s);
    /// returns leaf idx if node is okay
    int get_leaf_idx(node* leaf_node);
    /// returns a leaf node given an idx if node is okay
    node* get_leaf_node(int leaf_idx);

};

#endif // SUM_TREE_H
