#include "node.h"

node::node(double prio, node* parent, int idx, node* left_child, node* right_child):
    priority(prio),
    parent(parent),
    idx(idx),
    leaf_idx(-1), // -1 means not a leaf node
    left_child(left_child),
    right_child(right_child)
{}

node::~node(){
    // delete children
    if(this->left_child != nullptr){
        delete this->left_child;
    }
    if(this->right_child != nullptr){
        delete  this->right_child;
    }
}

