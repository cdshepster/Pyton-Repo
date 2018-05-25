class BSTNode:

    def __init__(self, key=0, lc=None, rc=None):
        self.__key = key
        self.__rightChild = rc
        self.__leftChild = lc

    def getKey(self):
        return self.__key
    def setKey(self, k):
        self.__key = k

    def getRightChild(self):
        return self.__rightChild
    def setRightChild(self, rc):
        self.__rightChild = rc

    def getLeftChild(self):
        return self.__leftChild
    def setLeftChild(self, lc):
        self.__leftChild = lc

    def __str__(self):
        b = 'BSTNode(key=' + str(self.__key)
        if self.__leftChild != None:
            b += ', lc=+, '
        else:
            b += ', lc=NULL, '
        if self.__rightChild != None:
            b += 'rc=+'
        else:
            b += 'rc=NULL'
        b += ')'
        return b
