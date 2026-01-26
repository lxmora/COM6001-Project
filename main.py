import networkx as nx
import matplotlib.pyplot as plt



if __name__ == "__main__":
    G = nx.karate_club_graph()
    nx.draw(G, with_labels=True)
    plt.show()
    
