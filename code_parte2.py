from matplotlib import patches, pyplot as plt
import networkx as nx
import seaborn as sns


def create_graph_from_txt(graph_filename):
    G = nx.Graph()
    with open(graph_filename, 'r', encoding='utf-8') as txtfile:
        for line in txtfile:
            deputado1, deputado2, votacoes_iguais = line.strip().split(';')
            G.add_edge(deputado1, deputado2, weight=int(votacoes_iguais))
            G.nodes[deputado1]['weight'] = int(votacoes_iguais)
            G.nodes[deputado2]['weight'] = int(votacoes_iguais)
    return G

def get_number_votes(arq):
    G = nx.Graph()
    with open(arq, 'r', encoding='utf-8') as txtfile:
        for line in txtfile:
            deputado, partido, numero_votos = line.strip().split(';')
            G.add_node(deputado, partido=partido, numero_votos=int(numero_votos))
    return G

def load_politicians_data(politicians_filename):
    politicians_data = {}
    with open(politicians_filename, 'r', encoding='utf-8') as txtfile:
        for line in txtfile:
            nome, partido, _ = line.strip().split(';')
            politicians_data[nome] = partido
    return politicians_data

def filter_graph(G, politicians_data, parties=None):
    filtered_G = G.copy()

    if parties is not None:
        nodes_to_remove = [node for node in filtered_G.nodes if politicians_data[node] not in parties]
        filtered_G.remove_nodes_from(nodes_to_remove)

    return filtered_G

def normalize_weights(graph, politicians_data):
    normalized_graph = graph.copy()
    for u, v, data in normalized_graph.edges(data=True):
        if u in politicians_data.nodes():
            numero_votos = politicians_data.nodes[u]['numero_votos']
            normalized_graph[u][v]['weight'] = data['weight'] / numero_votos
    return normalized_graph

def apply_threshold(graph, threshold):
    thresholded_graph = graph.copy()
    edges_to_remove = [(u, v) for u, v, data in thresholded_graph.edges(data=True) if float(data['weight']) < float(threshold)]
    thresholded_graph.remove_edges_from(edges_to_remove)
    return thresholded_graph

def invert_weights(graph):
    inverted_graph = graph.copy()
    for u, v, data in inverted_graph.edges(data=True):
        data['weight'] = 1 - data['weight']
    return inverted_graph

def compute_betweenness_centrality(graph):
    betweenness_centrality = nx.betweenness_centrality(graph)
    return betweenness_centrality

def plot_centrality_graph(graph, centrality_values, year):
    nodes = list(graph.nodes())
    centrality = [centrality_values[node] for node in nodes]
    
    sorted_nodes = [node for _, node in sorted(zip(centrality, nodes))]
    sorted_centrality = sorted(centrality)
    
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_nodes, sorted_centrality)
    plt.xlabel("Deputados")
    plt.ylabel("Centralidade de Betweenness")
    plt.title("Centralidade de Betweenness dos Deputados (Ordem Crescente)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.savefig(f"centralidade_betweenness_ordenado{year}.png")
    plt.show()
    
#função para mapa de calor
def create_heatmap(graph, year):
    adjacency_matrix = nx.to_pandas_adjacency(graph, dtype=float)
    correlation_matrix = adjacency_matrix.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="hot")
    plt.title("Mapa de Calor da Correlação entre Deputados")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)# Rotacionar rótulos do eixo X para melhor visibilidade
    plt.tight_layout()
    
    plt.savefig(f"heatmap{year}.png")
    plt.show()

def plot_graph(graph, politicians_data, parties, year):
    pos = nx.spring_layout(graph, k=2.0, iterations=0, seed=20)
    color = {}
    
    party_colors = {
        "PT": "red",
        "PSOL": "green",
        "NOVO": "blue",
        "REPUBLICANOS": "grey",
        "MDB" : "yellow",
        "PMDB" : "yellow",
        "PL" : "pink",
    }
    
    for node in graph.nodes():
        partido = politicians_data[node]
        if partido in party_colors:
            color[node] = party_colors[partido]
        else:
            color[node] = "gray"  # Cor padrão para partidos não mapeados
    
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(graph, pos, nodelist=graph.nodes(), node_size=300, node_color=list(color.values()))
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.7)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    
    # Criar legenda com as cores dos partidos
    legend_labels = {party: f"{party}: {party_colors[party]}" for party in party_colors}
    plt.legend(handles=[patches.Patch(color=party_colors[party], label=legend_labels[party]) for party in party_colors], bbox_to_anchor=(1, 1))
    
    plt.title("Grafo de Relações de Votos entre Deputados")
    plt.tight_layout()
    nome_arquivo = f"grafo_plot_{year}"
    plt.savefig(nome_arquivo)
    plt.show()

# Exemplo de uso
if __name__ == "__main__":
    year = input("Defina o ano que deseja:\n")
    
    user_input = input("Digite os nomes dos partidos em separados por vírgula em maiusculo, exemplo PT, PSOL: ")
    user_party_names = [party.strip() for party in user_input.split(",")]
    parties = user_party_names
    
    threshold = input("Informe o percentual minimo de concord^ancia ( threshold ) ( ex . 0.9):")

    graph_txt_filename = f"graph{year}.txt"
    politicians_txt_filename = f"politicians{year}.txt"
    
    grafo_numero_votos = get_number_votes(politicians_txt_filename)
    
    G = create_graph_from_txt(graph_txt_filename)
    politicians_data = load_politicians_data(politicians_txt_filename)
    
    filtered_graph = filter_graph(G, politicians_data, parties=parties)
    normalized_graph = normalize_weights(filtered_graph, grafo_numero_votos)
    thresholded_graph = apply_threshold(normalized_graph, threshold)
    inverted_weight_graph = invert_weights(thresholded_graph)
    compute_betweenness_centrality_graph = compute_betweenness_centrality(inverted_weight_graph)
    
    #grafico da centralidade
    plot_centrality_graph(inverted_weight_graph, compute_betweenness_centrality_graph, year)
    
    #grafico do mapa de calor
    create_heatmap(normalized_graph, year)
    
    #plot do grafo
    plot_graph(thresholded_graph, politicians_data, parties, year)
    
    
    