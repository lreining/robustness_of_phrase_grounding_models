import benepar, spacy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

benepar.download("benepar_en3")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

def hierarchy_pos_digraph(G, root, width=1., vert_gap=0.1, vert_loc=0., xcenter=0.5, pos=None):

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

    class Node:
        def __init__(self, s, identifier):
            self.sentence = s
            self.identifier = identifier
            self.children = []

    def __init__(self, sentence):
        self.sentence = sentence
        doc = nlp(sentence)
        sent = list(doc.sents)[0]
        self.root = self.build_tree(sent)

    def get_max_height(self):
        counts = [0]
        final_counts = []
        nodes = [self.root]
        while len(nodes)!=0:
            c, n = counts.pop(), nodes.pop()
            if len(n.children)!=0:
                nodes += n.children
                counts += [c+1 for _ in n.children]
            else:
                final_counts.append(c)
        return max(final_counts)

    def get_sentence_parts_for_level(self,level):
        max_height = self.get_max_height()
        if level<0:
            level = max_height+level+1

        if level>self.get_max_height() or level<0:
            raise ValueError

        identifiers = []
        sentences = []
        nodes = [self.root]
        counts = [0]

        while len(nodes) != 0:
            c, n = counts.pop(0), nodes.pop(0)
            if len(n.children)!=0 and c+1<=level:
                nodes = n.children + nodes
                counts = [c+1 for _ in n.children] + counts
            else:
                if str(n.sentence) != ".":
                    sentences += [n.sentence]
                    identifiers += [n.identifier]
        return sentences,identifiers

    def build_tree(self, sent):
        if sent is None:
            return
        label = sent._.labels[0] if len(sent._.labels)!=0 else ""
        node = self.Node(sent, label)
        for child in sent._.children:
            child_node = self.build_tree(child)
            if child_node is not None:
                node.children.append(child_node)
        return node

    def show(self,figsize=None):
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
