#!/usr/bin/python

###############################################
# module: rand_bst.py
# description: code for HW09 w/ unit  tests
# Cole Shepherd
###############################################
#1. As the number of nodes in a binary search tree goes to infinity, what is the probability of a binary search tree being a list?
# 0.00000000000000000
# 2. As the number of nodes in a binary search tree goes to infinity, what is the probability of a binary search tree being balanced?
# 0.00000000000000000


from BSTNode import BSTNode
from BSTree import BSTree
import random
import matplotlib.pyplot as plt
import numpy as np

def gen_rand_bst(num_nodes, a, b):
    bst = BSTree()
    numList = []
    for item in range(0, num_nodes):
        num = random.randint(a, b)
        while num in numList:
            num = random.randint(a, b)
        numList.append(num)

        bst.insertKey(num)

    return bst

def estimate_list_prob_in_rand_bsts_with_num_nodes(num_trees, num_nodes, a, b):
    linearCount = 0
    listOfBST = []

    for x in range(0, num_trees):
        bst = gen_rand_bst(num_nodes, a, b)
        if bst.isList():
            linearCount = linearCount + 1
            listOfBST.append(bst)

    probability = float(linearCount) / num_trees

    return probability, listOfBST


def estimate_list_probs_in_rand_bsts(num_nodes_start, num_nodes_end, num_trees, a, b):
    d = {}
    for num_nodes in xrange(num_nodes_start, num_nodes_end+1):
        d[num_nodes] = estimate_list_prob_in_rand_bsts_with_num_nodes(num_trees, num_nodes, a, b)
    return d

def estimate_balance_prob_in_rand_bsts_with_num_nodes(num_trees, num_nodes, a, b):
    balanceCount = 0
    listOfBST = []

    for x in range(0, num_trees):
        bst = gen_rand_bst(num_nodes, a, b)
        if bst.isBalanced():
            balanceCount = balanceCount + 1
            listOfBST.append(bst)

    probability = float(balanceCount) / num_trees

    return probability, listOfBST

def estimate_balance_probs_in_rand_bsts(num_nodes_start, num_nodes_end, num_trees, a, b):
    d = {}
    for num_nodes in xrange(num_nodes_start, num_nodes_end+1):
        d[num_nodes] = estimate_balance_prob_in_rand_bsts_with_num_nodes(num_trees, num_nodes, a, b)
    return d

def plot_rbst_lin_probs(num_nodes_start, num_nodes_end, num_trees):

    # list = estimate_list_probs_in_rand_bsts(num_nodes_start, num_nodes_end, num_trees, 1, 1000000)
    # prob = []
    #
    # for x in list:
    #     prob.append(x[0])
    #
    # plt.plot([range(num_nodes_start, num_nodes_end)], [probabilitys[0]], 'ro')
    # plt.axis([max(probabilitys[0]), min(probabilitys[0]), num_nodes_start, num_nodes_end])
    # plt.show()
    pass

def plot_rbst_balance_probs(num_nodes_start, num_nodes_end, num_trees):
    ## your code
    pass

### ========== UNIT TESTS =============

## unit_test_01 tests BSTNode constructor
##           5
##          /  \
##         3    10
def unit_test_01():
    r = BSTNode(key=5)
    lc = BSTNode(key=3)
    rc = BSTNode(key=10)
    print('root=%s, lc=%s, rc=%s' % (r, lc, rc))
    r.setLeftChild(lc)
    r.setRightChild(rc)
    assert ( r.getLeftChild().getKey()   == 3 )
    assert ( r.getRightChild().getKey() == 10 )

## unit_test_01() contstructs two bst's.
## bst
##        10
##       /  \
##      3   20
##
## bst2
##        5
##       /
##     3
##       \
##        4
def unit_test_02():
    bst = BSTree()
    bst.insertKey(10)
    bst.insertKey(3)
    bst.insertKey(20)
    assert ( bst.isBalanced() )
    assert ( bst.heightOf() == 1 )
    assert ( not bst.isList() )
    print('displaying bst')
    bst.displayInOrder()
    print('-------')

    bst2 = BSTree()
    bst2.insertKey(5)
    bst2.insertKey(3)
    bst2.insertKey(4)
    assert ( not bst2.isBalanced() )
    assert ( bst2.heightOf() == 2 )
    assert ( bst2.isList() )
    print('displaying bst2')
    bst2.displayInOrder()

def unit_test_03():
    rbst = gen_rand_bst(5, 0, 10)
    print('root=' + str(rbst.getRoot()))
    rbst.displayInOrder()
    print('is list? = ' + str(rbst.isList()))
    print('height = ' + str(rbst.heightOf()))
    print('is bal? = ' + str(rbst.isBalanced()))

def unit_test_04():
    print(estimate_list_prob_in_rand_bsts_with_num_nodes(100, 5, 0, 1000))

def unit_test_05():
    d = estimate_list_probs_in_rand_bsts(5, 20, 1000, 0, 1000000)
    for k, v in d.iteritems():
        print('probability of linearity in rbsts with %d nodes = %f' % (k, v[0]))

def unit_test_06(from_num_nodes, upto_num_nodes):
    d = estimate_list_probs_in_rand_bsts(from_num_nodes, upto_num_nodes, 1000, 0, 1000000)
    for k, v in d.iteritems():
        print('probability of linearity in rbsts with %d nodes = %f' % (k, v[0]))

def unit_test_07(num_nodes_start, num_nodes_end):
    d = estimate_balance_probs_in_rand_bsts(num_nodes_start, num_nodes_end, 1000, 0, 1000000)
    for k, v in d.iteritems():
        print('probability of balance in rbsts with %d nodes = %f' % (k, v[0]))



    
if __name__ == '__main__':
    plot_rbst_lin_probs(1, 5, 10)



