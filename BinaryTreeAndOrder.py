class BTree:
    def __init__(self,value):
        self.left = None
        self.data = value
        self.right = None
        self.parent = None

    def insertleft(self, value):
        self.left = BTree(value)
        self.left.parent = self
        return self.left

    def insertright(self, value):
        self.right = BTree(value)
        self.right.parent = self
        return self.right

    def show(self):
        print(self.data)


def preorder(node):
    if node.data:
        node.show()
        if node.left:
            preorder(node.left)
        if node.right:
            preorder(node.right)


def inorder(node):
    if node.data:
        if node.left:
            inorder(node.left)
        node.show()
        if node.right:
            inorder(node.right)


def postorder(node):
    if node.data:
        if node.left:
            postorder(node.left)
        if node.right:
            postorder(node.right)
        node.show()


root = BTree('R')
a = root.insertleft('A')
b = root.insertright('B')
c = a.insertleft('C')
d = a.insertright('D')
e = b.insertleft('E')

preorder(root)
inorder(root)
postorder(root)
