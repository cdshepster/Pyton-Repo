
#############################
# module: BSTree.py
# Cole Shepherd
#############################

from BSTNode import BSTNode

class BSTree:

    def __init__(self, root=None):
        self.__root = root
        if root==None:
            self.__numNodes = 0
        else:
            self.__numNodes = 1

    def getRoot(self):
        return self.__root

    def getNumNodes(self):
        return self.__numNodes

    def isEmpty(self):
        return self.__root == None

    def hasKey(self, key):
        if self.isEmpty():
            return False
        else:
            currNode = self.__root
            while currNode != None:
                if currNode.getKey() == key:
                    return True
                elif key < currNode.getKey():
                    currNode = currNode.getLeftChild()
                elif key > currNode.getKey():
                    currNode = currNode.getRightChild()
                else:
                    raise Exception('hasKey: ' + str(key))
            return False

    def insertKey(self, key):
        if self.isEmpty():
            self.__root = BSTNode(key=key)
            self.__numNodes += 1
            return True
        elif self.hasKey(key):
            return False
        else:
            currNode = self.__root
            parNode = None
            while currNode != None:
                parNode = currNode
                if key < currNode.getKey():
                    currNode = currNode.getLeftChild()
                elif key > currNode.getKey():
                    currNode = currNode.getRightChild()
                else:
                    raise Exception('insertKey: ' + str(key))
            if parNode != None:
                if key < parNode.getKey():
                    parNode.setLeftChild(BSTNode(key=key))
                    self.__numNodes += 1
                    return True
                elif key > parNode.getKey():
                    parNode.setRightChild(BSTNode(key=key))
                    self.__numNodes += 1
                    return True
                else:
                    raise Exception('insertKey: ' + str(key))
            else:
                raise Exception('insertKey: parNode=None; key= ' + str(key))
     
    def __heightOf(self, currnode):
        if currnode == None:
            return -1
        elif currnode.getLeftChild() == None and currnode.getRightChild() == None:
            return 0

        left = self.__heightOf(currnode.getLeftChild())
        right = self.__heightOf(currnode.getRightChild())
        return max(left, right) + 1

    def heightOf(self):
        return self.__heightOf(self.__root)

    def __isBalanced(self, currnode):
        leftChild = currnode.getLeftChild()
        rightChild = currnode.getRightChild()

        leftChildHeight = self.__heightOf(leftChild)
        rightChildHeight = self.__heightOf(rightChild)

        if abs(leftChildHeight - rightChildHeight) <= 1:
            return True
        else:
            return False

    def isBalanced(self):
        return self.__isBalanced(self.__root)

    def __displayInOrder(self, currnode):
        if currnode == None:
            print('NULL')
        else:
            self.__displayInOrder(currnode.getLeftChild())
            print(str(currnode))
            self.__displayInOrder(currnode.getRightChild())

    def displayInOrder(self):
        self.__displayInOrder(self.__root)

    def isList(self):
        verify = self.heightOf() + 1
        if verify == self.getNumNodes():
            return True
        else:
            return False