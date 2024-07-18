import benepar, spacy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# prepare language model and disable some parts not relevant to our research to
# make it faster

benepar.download("benepar_en3")
nlp = spacy.load("en_core_web_sm",disable=['ner', 'tok2vec'])

@spacy.Language.component("disable_sentence_segmentation")
def disable_sentence_segmentation(doc):
    for token in doc:
        token.is_sent_start = False
    return doc
nlp.add_pipe('disable_sentence_segmentation', before='parser')

nlp.add_pipe("benepar", config={"model": "benepar_en3"})

def hierarchy_pos_digraph(G, root, width=1., vert_gap=0.1, vert_loc=0., xcenter=0.5, pos=None):
    """
    Computes the position of each node in a hierarchical directed graph.

    Parameters:
    - G (networkx.DiGraph): The input graph.
    - root: The root node of the graph.
    - width (float): The width of the graph.
    - vert_gap (float): The vertical gap between nodes.
    - vert_loc (float): The vertical location of the root node.
    - xcenter (float): The x-coordinate of the root node.
    - pos (dict): The dictionary containing the positions of the nodes.

    Returns:
    - dict: The dictionary containing the positions of the nodes in the graph.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))

    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = hierarchy_pos_digraph(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap,
                                        xcenter=nextx, pos=pos)
    return pos
class SyntacticTree:
    """
    Represents a syntactic tree for a given sentence.

    Attributes:
    - sentence (str): The input sentence.
    - root (Node): The root node of the syntactic tree.

    Methods:
    - get_widths(): Returns a list of the number of children for each node in the tree.
    - get_max_height(): Returns the maximum height of the tree.
    - get_heights(): Returns a list of the heights of each node in the tree.
    - get_sentence_nodes_for_level(level): Returns a list of nodes at the specified level in the tree.
    - get_sentence_parts_for_level(level): Returns a tuple of lists containing the sentences and identifiers of nodes at the specified level in the tree.
    - build_tree(sent): Recursively builds the syntactic tree for a given sentence.
    - show(figsize): Visualizes the syntactic tree.
    """

    class Node:
        """
        Represents a node in a syntactic tree.

        Attributes:
        - sentence (str): The sentence associated with the node.
        - identifier (str): The identifier of the node.
        - children (list): The list of child nodes.
        """

        def __init__(self, s, identifier):
            self.sentence = str(s)
            self.identifier = str(identifier)
            self.children = []

    def __init__(self, sentence):
        """
        Initializes a SyntacticTree object with the given sentence.

        Parameters:
        - sentence (str): The input sentence.

        Returns:
        None
        """
        self.sentence = sentence
        doc = nlp(sentence)
        self.root = self.build_tree(list(doc.sents)[0])

    def get_widths(self):
        counts = []
        nodes = [self.root]
        while len(nodes)!=0:
            n = nodes.pop()
            if len(n.children) !=0:
                nodes += n.children
                counts += [len(n.children)]
        return counts

    def get_max_height(self):
        """
        Returns the maximum height of the syntactic tree.

        Returns:
        - int: The maximum height of the syntactic tree.
        """
        return np.max(self.get_heights())

    def get_heights(self):
        counts = [0]
        final_counts = []
        nodes = [self.root]
        while len(nodes) != 0:
            c, n = counts.pop(), nodes.pop()
            if len(n.children) != 0:
                nodes += n.children
                counts += [c + 1 for _ in n.children]
            else:
                final_counts.append(c)
        return final_counts
    
    def get_sentence_nodes_for_level(self, level):
        """
        Retrieves the sentence nodes at the specified level in the syntactic tree.

        Parameters:
        - level (int): The level of the tree to retrieve the sentence nodes from. 
                         A positive value represents the level from the root node, 
                         while a negative value represents the level from the deepest node.

        Returns:
        - list: A list of sentence nodes at the specified level.

        Raises:
        - ValueError: If the specified level is out of range.
        """
        max_height = self.get_max_height()
        if level < 0:
            level = max_height + level + 1


        if level > max_height or level < 0:
            print(level)
            print(max_height)
            raise ValueError


        final_nodes = []
        nodes = [self.root]
        counts = [0]

        while len(nodes) != 0:
            c, n = counts.pop(0), nodes.pop(0)
            if len(n.children) != 0 and c + 1 <= level:
                nodes = n.children + nodes
                counts = [c + 1 for _ in n.children] + counts
            else:
                if str(n.sentence) != ".":
                    final_nodes += [n]
        return final_nodes

    def get_sentence_parts_for_level(self,level):
        """
        Retrieves the sentences and identifiers for a given level in the syntactic tree.

        Parameters:
        - level (int): The level of the syntactic tree to retrieve the sentences and identifiers from.

        Returns:
        - tuple: A tuple containing two lists. The first list contains the sentences at the given level, and the second list contains the identifiers for each sentence.
        """
        nodes = self.get_sentence_nodes_for_level(level)
        sentences = list(map(lambda n:n.sentence, nodes))
        identifiers = list(map(lambda n:n.identifier, nodes))
        return sentences,identifiers

    def build_tree(self, sent):
        """
        Builds a syntactic tree from a given sentence.

        Parameters:
        - sent: The sentence object to build the tree from.

        Returns:
        - Node: The root node of the syntactic tree.

        """
        if sent is None:
            return None
        label = sent._.labels[0] if len(sent._.labels)!=0 else ""
        node = self.Node(sent, label)
        for child in sent._.children:
            child_node = self.build_tree(child)
            if child_node is not None:
                node.children.append(child_node)
        return node


    def show(self,figsize=None):
        """Visualize the syntactic tree"""
        figsize = figsize if figsize is not None else (10,10)
        plt.figure(figsize=figsize)
        G = nx.DiGraph()
        nodes = [self.root]
        ids = [0]

        counter = 0
        G.add_node(counter, name_vis=self.root.identifier)

        while len(nodes) != 0:
            node = nodes.pop()
            current_id = ids.pop()
            children = node.children
            if len(children) != 0:
                nodes += children
                children = [c.identifier if len(c.children)!= 0 else c.sentence for c in children]
            for c in children:
                counter += 1
                ids.append(counter)
                G.add_node(counter, name_vis=c)
                G.add_edge(current_id, counter)

        pos = hierarchy_pos_digraph(G, 0)
        labels = {n: (G.nodes[n]['name_vis']) for n in G.nodes}

        leaves = [n for n in G.nodes if len(list(G.neighbors(n)))==0]
        pos_leaves = {n:pos[n] for n in leaves}


        nodes = [n for n in G.nodes if len(list(G.neighbors(n)))!=0]
        pos_nodes = {n:pos[n] for n in nodes}


        nx.draw(G, with_labels=True, pos=pos,font_size="x-small", node_size=200, node_color='None',labels=labels)
        nx.draw_networkx_nodes(G, pos=pos_nodes,nodelist=nodes, edgecolors="black", node_color="skyblue", node_size=200)
        nx.draw_networkx_nodes(G, pos=pos_leaves, nodelist=leaves, edgecolors="None", node_color="None", node_size=200)
        plt.show()

