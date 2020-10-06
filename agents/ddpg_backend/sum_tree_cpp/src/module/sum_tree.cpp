#include <math.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "sum_tree.h"
#include <sstream>

sum_tree::sum_tree(int buffer_size, double init_priorities):
    buffer_size(buffer_size),
    n_nodes(2 * int(pow(2, ceil(log2(buffer_size)))) - 1),
    root(nullptr)
{
    if(2*this->buffer_size-1 != this->n_nodes){
        std::cout << "Stretching the number of nodes from " << 2*this->buffer_size-1 <<
                     " to " << this->n_nodes << ". This behavior is intended to get a full binary tree." << std::endl;
    }
    // initialize the tree
    // Note: all priorities are set to zero at the beginning
    init_tree(init_priorities);
}

sum_tree::~sum_tree(){
    // each node deletes its children
    delete this->root;
}

void sum_tree::init_tree(double init_priority){
    // check if there exists already a tree, if yes delete it first
    if(this->root != nullptr){
        std::cout << "Deleting old tree before creating new one ..." << std::endl;
        delete this->root;
        tree.clear();
        leaf_nodes.clear();
    }
    // create root node and add to tree
    this->root = new node(0.0, nullptr, 0);
    this->tree.push_back(root);

    // given the number of tree nodes we can initialize the sum tree
    int i = 0;
    int depth = ceil(log2(this->buffer_size));
    int curr_depth = 0;
    while(int(this->tree.size()) < this->n_nodes)
    {
        node* parent = tree[i];
        parent->left_child = new node(0.0, parent, tree.size());
        this->tree.push_back(parent->left_child);
        parent->right_child = new node(0.0, parent, tree.size());
        this->tree.push_back(parent->right_child);
        i++;

        // lets check if we are currently adding leaf nodes
        curr_depth = ceil(log2((double(this->tree.size()) + 1.f)/2.f));
        if(curr_depth == depth){
            this->leaf_nodes.push_back(parent->left_child);
            parent->left_child->leaf_idx = this->leaf_nodes.size() - 1;
            this->leaf_nodes.push_back(parent->right_child);
            parent->right_child->leaf_idx = this->leaf_nodes.size() - 1;
            // check, if we need to init the priorities to other than 0
            if(init_priority != 0.0 && parent->left_child->leaf_idx < this->buffer_size){
                update(parent->left_child->leaf_idx, init_priority);
            }
            if(init_priority != 0.0 && parent->right_child->leaf_idx < this->buffer_size){
                update(parent->right_child->leaf_idx, init_priority);
            }
        }
    }
}

void sum_tree::update(int leaf_idx, double priority){
    node* curr_node = get_leaf_node(leaf_idx);
    double change = priority - curr_node->priority;
    curr_node->priority = priority;
    // propagate the priority back to the root node
    propagate(curr_node->parent, change);
}

void sum_tree::update_mult(pybind11::array_t<int> leaf_idxs, pybind11::array_t<double> priorities){
    if(priorities.size() != leaf_idxs.size()){
        throw std::range_error("Length of priorities and leaf_idxs do not match!");
    }
    for(unsigned int i = 0; i < priorities.size(); i++){
        update(leaf_idxs.at(i), priorities.at(i));
    }
}

pybind11::tuple sum_tree::get(double s){
    node* leaf_node_ptr = retrieve(this->root, s);
    int data_idx = get_leaf_idx(leaf_node_ptr);
    double priority = get_leaf_node(data_idx)->priority;
    return pybind11::make_tuple(priority, data_idx);
}

pybind11::list sum_tree::get_leaf_priorities(int start, int end){
    // sanity check
    if(start < 0 || start >= end || end > buffer_size){
        std::stringstream err;
        err << "start is " << start << " and end is " << end <<
               "! This hurts restraints: start < 0 || start >= end || end > buffer_size";
        throw std::range_error(err.str());
    }
    pybind11::list priorities;
    for(int i = start; i < end; i++){
        priorities.append(this->leaf_nodes[i]->priority);
    }
    return priorities;
}

double sum_tree::total(){
    return root->priority;
}

int sum_tree::size(){
    return this->tree.size();
}

int sum_tree::size_leaves(){
    return this->leaf_nodes.size();
}

void sum_tree::propagate(node* node_ptr, double change){
    node_ptr->priority = node_ptr->priority + change;
    if(node_ptr->parent != nullptr){
        propagate(node_ptr->parent, change);
    }
}

node* sum_tree::retrieve(node* node_ptr, double s){
    // check if we are at leave node
    if(node_ptr->leaf_idx != -1){
        return node_ptr;
    }

    if(s <= node_ptr->left_child->priority){
        return retrieve(node_ptr->left_child, s);
    }
   else{
       return retrieve(node_ptr->right_child, s - node_ptr->left_child->priority);
   }

}

int sum_tree::get_leaf_idx(node* leaf_node){
    if(leaf_node->leaf_idx+1 > this->buffer_size && leaf_node->leaf_idx != -1){
        std::stringstream err;
        err << "Requesting a leaf_idx greater than the buffer_size! Total prioritiy is " << this->total();
       throw std::out_of_range(err.str());
    }
    else if(leaf_node->leaf_idx == -1){
        throw std::invalid_argument("Requested node isn't a leaf node!");
    }
    else{
        return leaf_node->leaf_idx;
    }
}

node* sum_tree::get_leaf_node(int leaf_idx){
    if(leaf_idx+1 > this->buffer_size && leaf_idx >= 0){
       throw std::out_of_range("Requesting a leaf_idx greater than the buffer_size!");
    }
    else if(leaf_idx < 0){
        throw std::invalid_argument("Requested leaf node idx is smaller than 0!");
    }
    else{
        return this->leaf_nodes[leaf_idx];
    }
}

