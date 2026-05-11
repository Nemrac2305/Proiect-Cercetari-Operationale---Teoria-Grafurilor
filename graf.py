import networkx as nx
import matplotlib.pyplot as plt


G = nx.DiGraph()

node_map = {
    "x1": "Militari",
    "x2": "Lujerului",
    "x3": "Crângași",
    "x4": "Răzoare",
    "x5": "Basarab",
    "x6": "Eroilor",
    "x7": "Arcul de Triumf",
    "x8": "Victoriei",
    "x9": "Barbu Văcărescu",
    "x10": "C. de Gaulle",
    "x11": "Pipera"
}

pos = {
    "x1": (0, 0),
    "x2": (8, 0),
    "x3": (13, 6),
    "x4": (13, -6),
    "x5": (20, 4),
    "x6": (20, -4),
    "x7": (20, 10),
    "x8": (27, 0),
    "x9": (31, 7),
    "x10": (34, 0),
    "x11": (41, 0)
}

edges = [
    ("x1", "x2", 6000), ("x2", "x3", 3000), ("x3", "x5", 2000),
    ("x3", "x7", 2000), ("x2", "x4", 2500), ("x4", "x6", 2000),
    ("x6", "x8", 2500), ("x5", "x8", 2500), ("x7", "x10", 2500),
    ("x7", "x9", 1500), ("x8", "x10", 4000), ("x9", "x11", 2500),
    ("x10", "x11", 3000)
]

for u, v, cap in edges:
    G.add_edge(u, v, capacity=cap)

plt.figure(figsize=(26, 15))

nx.draw_networkx_nodes(G, pos, node_size=9000, node_color='skyblue', edgecolors='black', linewidths=2.5)

nx.draw_networkx_labels(G, pos, font_size=20, font_weight='bold')

label_pos = {k: (v[0], v[1]-1.5) for k, v in pos.items()}
nx.draw_networkx_labels(G, label_pos, labels=node_map, font_size=15, font_color='black', font_weight='bold', verticalalignment='top')

other_edges = [e for e in G.edges() if e != ("x1", "x2")]
nx.draw_networkx_edges(G, pos, edgelist=[("x1", "x2")], width=3.5, arrowsize=50, edge_color='#2c3e50', connectionstyle="arc3,rad=0")
nx.draw_networkx_edges(G, pos, edgelist=other_edges, width=3.5, arrowsize=50, edge_color='#2c3e50', connectionstyle="arc3,rad=0.06")

edge_labels = {(u, v): f"{G[u][v]['capacity']}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=18,
    font_color='red',
    font_weight='bold',
    rotate=False,
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.3)
)

plt.title("Rețeaua Capacităților de Trafic București (Model x1 -> x11)", fontsize=28, fontweight='bold', pad=60)
plt.axis('off')
plt.margins(0.2)
plt.tight_layout()
plt.show()