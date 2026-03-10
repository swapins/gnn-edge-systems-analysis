import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

# --- Configuration ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Professional Color Palette
COLORS = {
    'box_edge': '#2c3e50',
    'box_face': '#ffffff',
    'node': '#d35400',
    'edge': '#7f8c8d',
    'text': '#2c3e50'
}

def draw_styled_box(ax, x, y, w, h, title, subtitle=None):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=COLORS['box_edge'],
        facecolor=COLORS['box_face'],
        zorder=1
    )
    ax.add_patch(rect)

    # Title - Anchored to top-middle of box
    ax.text(
        x + w/2, y + h - 0.08, 
        title,
        ha="center", va="top",
        fontsize=12, weight="bold",
        color=COLORS['text'],
        zorder=5
    )

    # Subtitle - Anchored to bottom-middle of box
    if subtitle:
        ax.text(
            x + w/2, y + 0.1,
            subtitle,
            ha="center", va="bottom",
            fontsize=10, style="italic",
            color=COLORS['text'],
            zorder=5
        )

def run_pipeline_viz():
    # Increase figsize slightly to give elements room to breathe
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 1. Draw Boxes (increased spacing)
    draw_styled_box(ax, 0.05, 0.25, 0.22, 0.5, "Transcriptomic Data", "Gene Expression\nMatrix ($X$)")
    draw_styled_box(ax, 0.39, 0.25, 0.22, 0.5, "PPI Network", "Graph Structure ($G$)")
    draw_styled_box(ax, 0.73, 0.25, 0.22, 0.5, "Constraint-Aware\nGNN Training")

    # 2. Draw Network (Centered in Middle Box)
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G, seed=42)
    
    # Scale nodes to fit strictly within the box boundaries
    for p in pos:
        pos[p] = [
            pos[p][0] * 0.07 + 0.5,      # X center of middle box
            pos[p][1] * 0.09 + 0.5       # Y center of middle box
        ]

    nx.draw_networkx_edges(G, pos, ax=ax, width=0.6, edge_color=COLORS['edge'], alpha=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color=COLORS['node'], linewidths=0.5, edgecolors='white')

    # 3. Draw Connecting Arrows (Adjusted lengths to avoid touching boxes)
    arrow_props = dict(arrowstyle="-|>", mutation_scale=15, color=COLORS['box_edge'], lw=1.5)
    ax.annotate("", xy=(0.38, 0.5), xytext=(0.28, 0.5), arrowprops=arrow_props)
    ax.annotate("", xy=(0.72, 0.5), xytext=(0.62, 0.5), arrowprops=arrow_props)

    # 4. Math Equation (Carefully centered in the Right Box)
    # Using a smaller fontsize and split lines to prevent overflow
    equation = (
        r"$\mathcal{L}_{total} = \mathcal{L}_{pred} +$" + "\n" +
        r"$\lambda (\mathcal{L}_{latency} + \mathcal{L}_{comp})$"
    )
    ax.text(
        0.84, 0.48,
        equation,
        ha="center", va="center",
        fontsize=10,
        color=COLORS['text'],
        zorder=10
    )

    plt.savefig("edge_gnn_fixed.png", bbox_inches="tight", dpi=600)
    plt.show()

if __name__ == "__main__":
    run_pipeline_viz()