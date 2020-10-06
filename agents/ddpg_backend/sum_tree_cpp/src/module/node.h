#ifndef NODE_H
#define NODE_H


class node
{
public:
    double priority;
    node* parent;
    int idx;
    int leaf_idx;
    node* left_child;
    node* right_child;

    node(double prio, node* parent, int idx, node* left_child=nullptr, node* right_child=nullptr);
    ~node();

};

#endif // NODE_H
