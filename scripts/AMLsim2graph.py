import pandas as pd
import networkx as nx
import numpy as np
import sys

'''
sys.path.insert(1, '../packages')
from   utils.graphConversion import __conversion__
import basicTests.basictest as bt
import localisation.localisationMeasures as lm
import netemdMethods.netEMDmeasures as nemdM #not needed
import pathFinder.pathFinder as pf
try:
    import community
except:
    print('Community detection package not found')
'''

sys.path.insert(1, '../packages/Elliot19_other_repo')
import basicTests.basictest as bt


sys.path.insert(1, '../packages/Elliot19_reproduce')
from basic_test import basic_features
from com_detection import community_detection
from utils import generate_null_models, get_parameters

def read_AMLsim_and_compute_graph(accounts_dir, transactions_dir):
    # Read in the CSV files
    accounts_df = pd.read_csv(accounts_dir)
    transactions_df = pd.read_csv(transactions_dir)

    # Create a directed graph
    graph = nx.MultiDiGraph()   
    graph2 = nx.DiGraph()
    graph3 = nx.DiGraph()

    # Add nodes to the graph using the accounts DataFrame
    for index, row in accounts_df.iterrows():
        graph.add_node(row["acct_id"], initial_deposit=row["initial_deposit"])
        graph2.add_node(row["acct_id"], initial_deposit=row["initial_deposit"])
        graph3.add_node(row["acct_id"])


    # Add edges to the graph using the transactions DataFrame
    for index, row in transactions_df.iterrows():
        graph.add_edge(row["orig_acct"], row["bene_acct"], base_amt=row["base_amt"], is_sar=row["is_sar"])

    # Print out some information about the graph
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    graph2.remove_nodes_from(list(nx.isolates(graph)))
    graph3.remove_nodes_from(list(nx.isolates(graph)))
    graph.remove_nodes_from(list(nx.isolates(graph))) #remove last lol otherwise no bene
    print("Number of nodes (after removing isolated nodes):", graph.number_of_nodes())


    graph = nx.convert_node_labels_to_integers(graph, first_label=0) #reorder indices due to the missing nodes (because we removed isolated nodes thus leaving behind non-continous indicies)
    graph2 = nx.convert_node_labels_to_integers(graph2, first_label=0)
    graph3 = nx.convert_node_labels_to_integers(graph3, first_label=0)

    for u in graph.nodes:
        graph2.nodes[u]["is_sar"] = False

    for u in graph.nodes: #loop through nodes. Current node is u
        tsc_amts = []
        for v in graph.predecessors(u):#loop though in degree nodes v
            ### WRONG ###
            #e_v2u = graph.in_edges(v, data=True) #edges from 
            e_v2u = [(s, t, d) for (s, t, d) in graph.edges(v, data=True) if t == u]
            #print(e_v2u)
            #print(e_v2u)
            edge_tsc_amts = []
            is_sar = False
            if e_v2u:
                for edge in e_v2u: #for each edge from v to u
                    
                    src, dest, attrs = edge
                    edge_tsc_amts.append(attrs["base_amt"])
                    if attrs["is_sar"] == True:
                        is_sar = True
                
                #get stats on edges
                v2u_total = np.sum(edge_tsc_amts)
                v2u_mean = np.mean(edge_tsc_amts)
                v2u_min = np.min(edge_tsc_amts)
                v2u_max = np.max(edge_tsc_amts)
                v2u_std = np.std(edge_tsc_amts)
                v2u_is_sar = is_sar
                v2u_count = len(edge_tsc_amts)

                graph2.add_edge(v, u, total_tsc_amt=v2u_total, mean_tsc_amt=v2u_mean, min_tsc_amt=v2u_min, max_tsc_amt=v2u_max, std_tsc_amt=v2u_std, num_of_tsc=v2u_count, contains_is_sar_tsc=v2u_is_sar)
                graph3.add_edge(v, u, weight=int(v2u_total))

                if v2u_is_sar == True:
                    graph2.nodes[u]["is_sar"] = True
                    graph2.nodes[v]["is_sar"] = True
                

    print("Number of nodes:", graph2.number_of_nodes())
    print("Number of edges:", graph2.number_of_edges())
    

    ### BASIC TEST MODULE:
        #TWO METHOD: 
            #METHOD 1: DIMOS REPO OF ELLIOT19 CODE: much slower. Needs computation of null_sampels
        
            #METHOD 2: NEW ELLIOT19 CODE: faster but not the same for GAW, GAW10, GAW20 (not much difference)

    # METHOD 1
    '''
    G = graph3.copy()
    #G = basic_features(G, num_samples=1000)

    
    num_models = 1              # number of artificial graphs created
    num_nodes = 1000            # number of graph's nodes
    num_basic_mc_samples = 500  # number of replicas from the model, which 
                                # fixes the degrees and randomly shuffles weights
    num_references = 10         # WHAT
    num_null_models = 60        # WHAT

    print("Basic Features : Start")
    G = basic_features(G, num_samples=1000)
    print("Basic Features : Done")
    '''
    
    # METHOD 2
    
    ## Add netemd Statistics
    results = {}
    G = graph3.copy()
    '''
    G1 = __conversion__(G)
    G1[0]
    pathRes = pf.pathFinderRandomComparision(G1,beamwidth=100,maxLen=5,reps=50)
    results['Path of Size 5'] = {i:0 for i in range(len(G))}
    results['Path of Size 6'] = {i:0 for i in range(len(G))}
    for item in pathRes[0][2]:
        for x in item[0][1]:
            results['Path of Size 5'][x] += item[2]
    for item in pathRes[0][3]:
        for x in item[0][1]:
            results['Path of Size 6'][x] += item[2]
    '''

    #G = nx.convert_node_labels_to_integers(G, first_label=0) #reorder indices due to the missing nodes (because we removed isolated nodes thus leaving behind non-continous indicies)
    results['GAW']    = bt.strengthDegree(G,1000,0.05)[0][2]
    results['GAW10']  = bt.strengthDegree10(G,1000,0.05)[0][2]
    results['GAW20']  = bt.strengthDegree20(G,1000,0.05)[0][2]
    results['Std Degree']  = bt.getTotalDegree(G)[0][1]
    G2 = G.to_undirected()
    #coms=community.best_partition(G2)
    #results['communityDensity'] = bt.subNetworkDensityTest(G,coms,1000)[0][1]

    results['Feature Sum']={x:0 for x in G}
    for item in results:
        if item!='Feature Sum':
            for x in results[item]:
                results['Feature Sum'][x]+=results[item][x]
    # end of Basic tests

    # start of community detection
    num_models = 1              # number of artificial graphs created
    num_nodes = 1000            # number of graph's nodes
    num_basic_mc_samples = 500  # number of replicas from the model, which 
                                # fixes the degrees and randomly shuffles weights
    num_references = 10         # WHAT
    num_null_models = 60        # WHAT

    null_samples_whole, null_samples = generate_null_models(G, num_models=num_null_models, min_size=20)
    G = community_detection(G, null_samples, num_samples=20)


    # end of community detection
    


    for node in graph2.nodes():
        graph2.nodes()[node]["GAW"] = results['GAW'][node]
        graph2.nodes()[node]["GAW10"] = results['GAW10'][node]
        graph2.nodes()[node]["GAW20"] = results['GAW20'][node]
        graph2.nodes()[node]["Std Degree"] = results['Std Degree'][node]
        
        graph2.nodes()[node]["first_density"] = G.nodes()[node]['first_density']
        graph2.nodes()[node]["second_density"] = G.nodes()[node]['second_density']
        graph2.nodes()[node]["third_density"] = G.nodes()[node]['third_density']
        graph2.nodes()[node]["small_community"] = G.nodes()[node]['small_community']
        graph2.nodes()[node]["first_strength"] = G.nodes()[node]['first_strength']
        graph2.nodes()[node]["second_strength"] = G.nodes()[node]['second_strength']

    print(graph2)
    print(graph2.nodes()[0])

    nodes_df = pd.DataFrame.from_dict(dict(graph2.nodes(data=True)), orient='index')
    edges_df = pd.DataFrame([(u, v, d['total_tsc_amt'], d['mean_tsc_amt'], d['min_tsc_amt'], d['max_tsc_amt'], d['std_tsc_amt'], d['num_of_tsc'], d['contains_is_sar_tsc']) for u, v, d in graph2.edges(data=True)], columns=['source', 'target', 'total_tsc_amt','mean_tsc_amt', 'min_tsc_amt', 'max_tsc_amt', 'std_tsc_amt', 'num_of_tscs', 'contains_is_sar_tsc'])

    
    new_accounts_dir = accounts_dir[:-4]+"_digraph_form.csv"
    new_transactions_dir = transactions_dir[:-4]+"_digraph_form.csv"

    nodes_df.to_csv(new_accounts_dir)
    edges_df.to_csv(new_transactions_dir)

    print("Created new graph saved to", new_accounts_dir, new_transactions_dir)

    return graph2, G, results