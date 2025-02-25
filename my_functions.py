import json
import torch
import os
from torch_geometric.data import Data
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed
import random


## FUNZIONI PER ESTRAZIONE IPERGRAFI ###
def ipergrafiPC(data, thresholds):
    hyperedges_dict = {t: defaultdict(set) for t in thresholds}  # Usiamo set() per evitare duplicati

    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            corr_ij, _ = pearsonr(data[:, i], data[:, j])
            for t in thresholds:
                if abs(corr_ij) >= t:
                    hyperedges_dict[t][f"G{i+1}"].add(f"G{j+1}")
                    hyperedges_dict[t][f"G{j+1}"].add(f"G{i+1}")

    # Convertiamo in lista di liste per avere iperarchi
    for t in thresholds:
        hyperedges_dict[t] = [list(hyperedges_dict[t][k]) + [k] for k in hyperedges_dict[t]]

    return hyperedges_dict


def extract_hyperedges(input_file, output_file, type, thresholds):
    if not os.path.exists(input_file):
        print(f"Errore: il file {input_file} non esiste.")
        return

    print("Caricamento della matrice di espressione genica...")
    data, _ = read_expression_data(input_file)

    # estrazione degli iperarchi
    if type == 'PC':
        print("Estrazione degli iperarchi...")
        hyperedges_dict1 = ipergrafiPC(data, thresholds)

        # salvataggio degli iperarchi su file JSON
        print(f"Salvataggio degli iperarchi su {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(hyperedges_dict1, f, indent=4)

        print(f"Processo completato, iperarchi estratti usando: {type}")
        for t in thresholds:
            print(f"Soglia {t}: {len(hyperedges_dict1[t])} iperarchi estratti.")


## FUNZIONI PER SOTTO-IPERGRAFI ###
def read_expression_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Leggi i nomi dei geni dalla prima riga
    gene_names = lines[0].strip().split("\t")  # Assumiamo separatore tab

    # Leggi i dati numerici (saltando la prima riga)
    data = np.array([list(map(float, line.strip().split("\t"))) for line in lines[1:]])

    return data, gene_names


def load_hypergraph(json_file, threshold):
    with open(json_file, "r") as f:
        hyperedges_dict = json.load(f)

    if threshold in hyperedges_dict:
        hyperedges = hyperedges_dict[threshold]
    else:
        return [], []

    unique_genes = list(set(g for edge in hyperedges for g in edge))
    return hyperedges, unique_genes


def calculate_node_features(data, gene_names, unique_genes):
    features = {}
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

    for gene_name in unique_genes:
        if gene_name not in gene_to_idx:
            continue

        gene_idx = gene_to_idx[gene_name]
        gene_expression = data[:, gene_idx]

        if np.std(gene_expression) == 0:
            z_score = np.zeros_like(gene_expression)
        else:
            z_score = (gene_expression - np.mean(gene_expression)) / np.std(gene_expression)

        std_dev = np.std(gene_expression)
        q1 = np.percentile(gene_expression, 25)
        q3 = np.percentile(gene_expression, 75)

        features[gene_name] = np.array([z_score.mean(), std_dev, q1, q3])

    return features


def extract_positive_pairs(hyperedges):
    # crea tutte le possibili coppie di geni che sono presenti nello stesso iperarco
    positive_pairs = set()
    for hyperedge in hyperedges:
        if len(hyperedge) > 1:
            for u, v in combinations(hyperedge, 2):
                positive_pairs.add((min(u, v), max(u, v)))
    return positive_pairs


def process_pair(pair, hyperedges, unique_genes, expression_data, gene_names, node_features_dict, label):
    gene_a, gene_b = pair
    relevant_hyperedges = [h for h in hyperedges if gene_a in h or gene_b in h]

    subgraph_genes = set()
    for h in relevant_hyperedges:
        subgraph_genes.update(h)

    if len(subgraph_genes) < 2:
        return None

    subgraph_genes = list(subgraph_genes)
    sub_gene_to_idx = {gene: i for i, gene in enumerate(subgraph_genes)}

    edge_index = []
    for h in relevant_hyperedges:
        mapped_h = [sub_gene_to_idx[g] for g in h]
        for u, v in combinations(mapped_h, 2):
            edge_index.append([u, v])
    # edge_index: contiene le informazioni specificando quali geni sono connessi tra loro
    #             (che sono presenti negli iperarchi dove compare la coppia)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    x = np.zeros((len(subgraph_genes), 4))
    for gene, idx in sub_gene_to_idx.items():
        x[idx] = node_features_dict.get(gene, np.zeros(4))
    # x: ogni gene del sotto-ipergrafo con le sue informazioni

    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.float) #1

    return Data(x=x, edge_index=edge_index, y=y)


def create_subgraph_data(json_file, expression_data, gene_names, threshold):
    hyperedges, unique_genes = load_hypergraph(json_file, threshold)
    if not hyperedges:
        return None

    positive_pairs = extract_positive_pairs(hyperedges)
    node_features_dict = calculate_node_features(expression_data, gene_names, unique_genes)

    results = Parallel(n_jobs=-1)(
        delayed(process_pair)(pair, hyperedges, unique_genes, expression_data, gene_names, node_features_dict, 1) for
        pair in tqdm(positive_pairs))

    dataset = [data for data in results if data is not None]
    return dataset


def process_negative_pair(pair, hyperedges, data, gene_names):
    gene1, gene2 = pair
    relevant_hyperedges = [h for h in hyperedges if gene1 in h or gene2 in h]

    subgraph_genes = set()
    for h in relevant_hyperedges:
        subgraph_genes.update(h)

    if len(subgraph_genes) < 2:
        return None

    subgraph_genes = list(subgraph_genes)
    sub_gene_to_idx = {gene: i for i, gene in enumerate(subgraph_genes)}

    edge_index = []
    for h in relevant_hyperedges:
        mapped_h = [sub_gene_to_idx[g] for g in h]
        for u, v in combinations(mapped_h, 2):
            edge_index.append([u, v])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    node_features_dict = calculate_node_features(data, gene_names, subgraph_genes)
    x = np.zeros((len(subgraph_genes), 4))
    for gene, idx in sub_gene_to_idx.items():
        x[idx] = node_features_dict.get(gene, np.zeros(4))

    x = torch.tensor(x, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=torch.tensor([0], dtype=torch.float))


def generate_negative_subgraphs(hyperedges, data, gene_names, num_neg_samples):
    all_gene_pairs = set(combinations(gene_names, 2))
    positive_pairs = {pair for hyperedge in hyperedges for pair in combinations(hyperedge, 2)}
    negative_pairs = list(all_gene_pairs - positive_pairs)

    np.random.shuffle(negative_pairs)
    negative_pairs = negative_pairs[:num_neg_samples]

    results = Parallel(n_jobs=-1)(
        delayed(process_negative_pair)(pair, hyperedges, data, gene_names)
        for pair in tqdm(negative_pairs, desc="Generating negative subgraphs")
    )

    return [graph for graph in results if graph is not None]