import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
from collections import Counter, defaultdict
from itertools import combinations
from networkx.algorithms.community import girvan_newman
from cdlib import algorithms

# Generates a top 20 domains by amount

def top20():
    df = pd.read_csv('trackers.csv', thousands=',')
    top20 = df.nlargest(20, 'times_seen').sort_values('times_seen')

    # Plot top20
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')
    ax.barh(top20['tracker'], top20['times_seen'])

    # Set limit on x-axis for readablity
    max_val = top20['times_seen'].max()
    ax.set_xlim(0, max_val * 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')

    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_title('Top 20 Trackers by Times Observed', fontsize=16)
    ax.set_xlabel('Times observed',               fontsize=14)
    ax.set_ylabel('Tracker',                      fontsize=14)

    plt.tight_layout()
    fig.savefig('top20.png', facecolor='white', edgecolor='none')

# Generates a distribution graph of unique trackers per domain
def distribution():           
    urldf = pd.read_csv('urls_with_trackers.csv', usecols=['domain','trackers_id'])
    urldf['trackers_id'] = pd.to_numeric(urldf['trackers_id'], errors='coerce')
    urldf = urldf.dropna(subset=['trackers_id'])
    urldf['trackers_id'] = urldf['trackers_id'].astype(int)

    # plot distribution
    trackers_per_domain = urldf.groupby('domain')['trackers_id'].nunique()
    dist = trackers_per_domain.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')                                   
    ax.bar(dist.index, dist.values)

    # Limit x-axis for readability
    ax.set_xlim(0, 16)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')
    ax.tick_params(axis='both', which='major', labelsize=11) 

    ax.set_xlabel('Unique trackers',         fontsize=14)
    ax.set_ylabel('Number of domains',      fontsize=14)
    ax.set_title('Distribution of Trackers per Domain', fontsize=16)

    plt.tight_layout()
    fig.savefig('distribution.png', facecolor='white', edgecolor='none')
    print("Saved distribution plot")

def tracker_types():
    tracker_lookup = {}
    with open('trackers.csv', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tid = int(row['id'])
            except ValueError:
                continue
            tracker_lookup[tid] = row['tracker'].strip().lower()
    
    counts = Counter({'Dutch': 0, 'Non-Dutch': 0})
    with open('urls_with_trackers.csv', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid_str = row.get('trackers_id', '').strip()
            if not tid_str:
                continue
            try:
                tid = int(tid_str)
            except ValueError:
                continue
            
            tracker = tracker_lookup.get(tid, '')
            if tracker.endswith('.nl'):
                counts['Dutch'] += 1
            else:
                counts['Non-Dutch'] += 1
    
    print('Dutch: ', counts['Dutch'])
    print('Non Dutch: ', counts['Non-Dutch'])
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    
    plt.figure(figsize=(6,4), facecolor='#f5f5dc')
    plt.bar(labels, values)
    plt.xlabel('Tracker type')
    plt.ylabel('Number of tracker occurrences')
    plt.title('Dutch vs. Non-Dutch Trackers')
    plt.tight_layout()
    
    plt.savefig('types.png', facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    print("Saved types plot")

# Build network of common trackers and save weighted/unweighted distributions
def common_tracker():
    domain_trackers = defaultdict(set)
    with open('urls_filtered3.csv', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row['trackers_id'].strip()
            if not t:
                continue
            domain_trackers[row['domain']].add(t)

    # 2. Invert to tracker→domains
    tracker_domains = defaultdict(list)
    for domain, trackers in domain_trackers.items():
        for t in trackers:
            tracker_domains[t].append(domain)

    # 3. Accumulate edge weights
    edge_weights = defaultdict(int)
    for domains in tracker_domains.values():
        for u, v in combinations(domains, 2):
            edge_weights[(u, v)] += 1

    # 4. Build graph
    G = nx.Graph()
    G.add_nodes_from(domain_trackers)
    G.add_weighted_edges_from((u, v, w) for (u, v), w in edge_weights.items())
    
    
    # Unweighted degree distribution
    deg_sequence = [deg for _, deg in G.degree()]
    deg_counts = Counter(deg_sequence)
    degrees, counts = zip(*sorted(deg_counts.items()))
    plt.figure()
    plt.loglog(degrees, counts, marker='o', linestyle='None')
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('Degree Distribution (log–log)')
    plt.tight_layout()
    plt.savefig('degree_distribution.png')
    plt.close()

    # Weighted degree distribution
    wdeg_sequence = [wdeg for _, wdeg in G.degree(weight='weight')]
    wdeg_counts = Counter(wdeg_sequence)
    wdeg_values, wdeg_counts_vals = zip(*sorted(wdeg_counts.items()))
    plt.figure()
    plt.loglog(wdeg_values, wdeg_counts_vals, marker='o', linestyle='None')
    plt.xlabel('Weighted degree')
    plt.ylabel('Number of nodes')
    plt.title('Weighted Degree Distribution (log–log)')
    plt.tight_layout()
    plt.savefig('weighted_degree_distribution.png')
    plt.close()

    nx.write_gexf(G, "domains_graph.gexf")
    return G

def analyse_network(nodes_to_check: list[str] = None):
    G = nx.read_gexf('domains_graph.gexf')
    print('klaar met inlezen gexf')

    # 2. Centrality measures
    degree_centrality = nx.degree_centrality(G)
    print('klaar met degree centrality')
    betweenness_centrality = nx.betweenness_centrality(G)
    print('klaar met betweenness centrality')
    closeness_centrality = nx.closeness_centrality(G)
    print('klaar met closeness centrality')

    # 3. Network-level metrics
    # Average degree = sum of degrees / number of nodes
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print('klaar met average degree')
    # Density of the network
    density = nx.density(G)
    print('klaar met density')

    # 4. Community detection: Girvan-Newman (first split)
    gn_gen = girvan_newman(G)
    node_gn_split = next(gn_gen)
    node_gn_comms = [set(c) for c in node_gn_split]
    # 4b. Louvain (via CDlib)
    lv_node = algorithms.louvain(G)
    node_louvain_comms = [set(c) for c in lv_node.communities]
    print('klaar met girvan newman')

    # 5. Community detection: Louvain via CDlib
    L = nx.line_graph(G)
    # 5a. Louvain on edges
    lv_edge = algorithms.louvain(L)
    edge_louvain_comms = [set(c) for c in lv_edge.communities]
    # 5b. Girvan-Newman on edges (first split)
    edge_gn_gen = girvan_newman(L)
    edge_gn_split = next(edge_gn_gen)
    edge_gn_comms = [set(c) for c in edge_gn_split]
    print('klaar met louvain')

    # 6. Check specified nodes' community membership
    checked_nodes_comms = {}
    checked_nodes_same = None
    if nodes_to_check:
        for node in nodes_to_check:
            checked_nodes_comms[node] = None
            for idx, comm in enumerate(node_louvain_comms):
                if node in comm:
                    checked_nodes_comms[node] = idx
                    break
        found = [c for c in checked_nodes_comms.values() if c is not None]
        if found:
            checked_nodes_same = (len(set(found)) == 1)
    print('klaar met checking nodes')

    # 7. Return all results
    with open('Metrics.txt', 'w', encoding='utf-8') as f:
        # Network metrics
        f.write('=== Network Metrics ===\n')
        f.write(f'Average degree: {avg_degree:.6f}\n')
        f.write(f'Density: {density:.6f}\n\n')

        # Centrality measures
        f.write('=== Centrality Measures ===\n')
        f.write('Node\tDegreeCentrality\tBetweennessCentrality\tClosenessCentrality\n')
        for n in G.nodes():
            f.write(f"{n}\t{degree_centrality[n]:.6f}\t"
                    f"{betweenness_centrality[n]:.6f}\t"
                    f"{closeness_centrality[n]:.6f}\n")
        f.write('\n')

        # Node communities
        f.write('=== Node Girvan-Newman Communities ===\n')
        for idx, comm in enumerate(node_gn_comms):
            f.write(f'Community {idx}: {sorted(comm)}\n')
        f.write('\n')

        f.write('=== Node Louvain Communities ===\n')
        for idx, comm in enumerate(node_louvain_comms):
            f.write(f'Community {idx}: {sorted(comm)}\n')
        f.write('\n')

        # Edge communities
        f.write('=== Edge Louvain Communities ===\n')
        for idx, comm in enumerate(edge_louvain_comms):
            # nodes of L are edge-tuples from G
            edges = sorted(comm)
            f.write(f'Edge Community {idx}: {edges}\n')
        f.write('\n')

        f.write('=== Edge Girvan-Newman Communities ===\n')
        for idx, comm in enumerate(edge_gn_comms):
            f.write(f'Edge Community {idx}: {sorted(comm)}\n')
        f.write('\n')

        # Checked nodes
        if nodes_to_check:
            f.write('=== Checked Nodes Communities ===\n')
            for node, cid in checked_nodes_comms.items():
                f.write(f'Node {node}: Community {cid}\n')
            f.write(f'All in same community: {checked_nodes_same}\n')

trackers = ['fonts.googleapis.com', 'facebook.com', 'googletagmanager.com', 'instagram.com', 'google.com', 
            'gmpg.com', 'youtube.com', 'linkedin.com', 'twitter.com', 'cdn.shopify.com', 
            'stats.wp.com', 'google.nl', 'one.com', 'wordpress.org', 'pinterest.com', 
            'statcounter.com', 'unpkg.com', 'blogger.com', 'x.com', 'addtoany.com', ]

analyse_network(trackers)