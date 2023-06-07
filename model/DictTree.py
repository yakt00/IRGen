class TreeNode(object):
    def __init__(self):
        self.nodes = {}      
        self.is_leaf = False   
        self.count = 0          

    def insert(self,word):
        curr = self
        for c in word:
            if not curr.nodes.get(c,None):
                new_node = TreeNode()
                curr.nodes[c] = new_node
            curr = curr.nodes[c]
            curr.is_leaf = True
        self.count += 1
        return

    def insert_many(self,words):
        for word in words:
            self.insert(word)
        return

    def search(self,word):
        curr = self
        try:
            for c in word:
                curr = curr.nodes[c]
        except:
            return False
        return curr.is_leaf
