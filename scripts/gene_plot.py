import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

edges = [
("TP53","EGFR"),
("TP53","BRCA1"),
("TP53","KRAS"),
("EGFR","KRAS"),
("BRCA1","KRAS"),
("TP53","GENE_A"),
("EGFR","GENE_B"),
]

G.add_edges_from(edges)

pos = nx.spring_layout(G)

nx.draw(G,pos,node_size=500,with_labels=True)
plt.savefig("gene_importance_network.png",dpi=300)