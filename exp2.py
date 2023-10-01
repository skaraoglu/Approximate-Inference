import json
import random
import time
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from multiprocessing import Pool

class BayesianNetwork:
    def __init__(self, json_file):
        with open(json_file) as f:
            self.bn_data = json.load(f)

    def weighted_sample(self, evidence):
        sample = {}
        weight = 1.0

        for node in self.bn_data:
            if node in evidence:
                for prob in self.bn_data[node]['prob']:
                    if prob[0] == []:
                        weight *= prob[1] if evidence[node] == 1 else 1 - prob[1]
                sample[node] = evidence[node]
            else:
                parents = self.bn_data[node]['parents']
                parent_values = [sample[str(parent)] for parent in parents if str(parent) in sample]
                prob = random.random()
                for prob_list in self.bn_data[node]['prob']:
                    if prob_list[0] == parent_values:
                        sample[node] = 1 if prob < prob_list[1] else 0
        # Ensure that a sample is generated for each node
        for node in self.bn_data:
            if node not in sample:
                sample[node] = np.random.choice([0, 1])

        return sample, weight

    def likelihood_weighting(self, query_variable, evidence, N):
        # Initialize counts for each possible outcome of the query variable
        counts = {0: 0, 1: 0}

        for _ in range(N):
            # Generate a weighted sample
            X, weight = self.weighted_sample(evidence)

            counts[X[query_variable]] += weight

        # Normalize the counts to get probabilities
        total_count = sum(counts.values())
        return {key: value / total_count for key, value in counts.items()}
    
    def gibbs_ask(self, query_variable, evidence, N):
        # Initialize non-evidence variables to random values
        X = {var: np.random.choice([0, 1]) for var in self.bn_data if var not in evidence}
        X.update(evidence)

        counts = {0: 0, 1: 0}

        Z = [var for var in self.bn_data if var not in evidence]

        for _ in range(N):
            for Zi in Z:
                # Sample a value for Zi given its Markov blanket
                X[Zi] = self.sample_given_markov_blanket(Zi, X)
            counts[X[query_variable]] += 1

        # Normalize the counts to get probabilities
        total_count = sum(counts.values())
        return {key: value / total_count for key, value in counts.items()}

    def sample_given_markov_blanket(self, node, X):
        # Calculate the conditional probability of the node given its Markov blanket
        prob = self.calculate_conditional_probability(node, X)
        return 1 if np.random.random() < prob else 0

    def calculate_conditional_probability(self, node, X):
        # Calculate the conditional probability P(node | parents(node), children(node), parents(children(node)))
        parents = self.bn_data[node]['parents']
        parent_values = [X[str(parent)] for parent in parents]

        for prob_list in self.bn_data[node]['prob']:
            if prob_list[0] == parent_values:
                return prob_list[1]

        return 0.5  # Default probability if no match is found
    
    def metropolis_hastings(self, query_variable, evidence, N, p=0.5):
        # Initialize non-evidence variables to random values
        X = {var: np.random.choice([0, 1]) for var in self.bn_data if var not in evidence}
        X.update(evidence)

        counts = {0: 0, 1: 0}

        for _ in range(N):
            if np.random.random() < p:
                # Perform Gibbs sampling
                for Zi in X:
                    # Sample a value for Zi given its Markov blanket
                    X[Zi] = self.sample_given_markov_blanket(Zi, X)
            else:
                # Generate a random sample
                X, _ = self.weighted_sample(evidence)

            counts[X[query_variable]] += 1

        # Normalize the counts to get probabilities
        total_count = sum(counts.values())
        return {key: value / total_count for key, value in counts.items()}

def select_nodes_of_interest(json_file):
    # Load the Bayesian Network from the JSON file
    with open(json_file) as f:
        bn_data = json.load(f)

    # Calculate the number of parents and children for each node
    node_importance = {}
    for node, data in bn_data.items():
        num_parents = len(data['parents'])
        num_children = sum(node in data['parents'] for data in bn_data.values())
        node_importance[node] = (num_parents, num_children)

    # Sort the nodes by their importance (number of parents + number of children)
    sorted_nodes = sorted(node_importance, key=lambda x: (node_importance[x][0] + node_importance[x][1]), reverse=True)
    sorted_nodes_parents = sorted(node_importance, key=lambda x: node_importance[x][0], reverse=True)
    sorted_nodes_children = sorted(node_importance, key=lambda x: node_importance[x][1], reverse=True)

    # Select nodes based on the criteria
    nodes_of_interest = []
    for node_list in [sorted_nodes_parents, sorted_nodes_children, sorted_nodes]:
        for node in node_list:
            if node not in nodes_of_interest:
                nodes_of_interest.append(node)
                break

    # Find a root node (a node with no parents)
    for node, data in bn_data.items():
        if len(data['parents']) == 0 and node not in nodes_of_interest:
            nodes_of_interest.append(node)
            break

    # Find a leaf node (a node with no children)
    for node in bn_data:
        if all(node not in data['parents'] for data in bn_data.values()) and node not in nodes_of_interest:
            nodes_of_interest.append(node)
            break

    # Add two random nodes if they are not already in the list
    all_nodes = list(bn_data.keys())
    for _ in range(2):
        while True:
            node = random.choice(all_nodes)
            if node not in nodes_of_interest:
                nodes_of_interest.append(node)
                break

    return nodes_of_interest

def select_evidence_nodes(bn_file, nodes_of_interest):
    # Load the Bayesian Network from the JSON file
    with open(bn_file) as f:
        bn = json.load(f)
    # Get the list of nodes in the network
    nodes = list(bn.keys())

    # Remove nodes of interest from the list of nodes
    for node in nodes_of_interest:
        nodes.remove(node)

    # Select a root node (a node with no parents)
    root_node = next((node for node in nodes if not any(node in data['parents'] for data in bn.values())), None)

    # If no root node is found, select a random node
    if root_node is None:
        root_node = np.random.choice(nodes)

    # Remove the root node from the list of nodes
    nodes.remove(root_node)

    # Select a leaf node (a node that is not a parent of any other node)
    leaf_node = next((node for node in nodes if not any(node in data['parents'] for data in bn.values())), None)

    # If no leaf node is found, select a random node
    if leaf_node is None:
        leaf_node = np.random.choice(nodes)

    return root_node, leaf_node

def run_exact_inference(bn_file, node):
    query_var = node
    # Run the exact inference script and capture its output
    result = subprocess.run(['python', 'exact-inference.py', '-f', bn_file, '-q', str(query_var)], capture_output=True, text=True)

    # Parse the output to extract the time and probabilities
    lines = result.stdout.split('\n')
    time_line = next(line for line in lines if line.startswith('Time'))
    time = time_line.split(':')[1].strip().replace(' secs', '')

    p_false_line = next(line for line in lines if line.startswith('P_False'))
    p_false = float(p_false_line.split(':')[1].strip())

    p_true_line = next(line for line in lines if line.startswith('P_True'))
    p_true = float(p_true_line.split(':')[1].strip())

    return {'Time': time, 'P_False': p_false, 'P_True': p_true}

def run_method(args):
    bn_file, node, method, p, evidence = args
    bn = BayesianNetwork(bn_file)
    starttime = time.time()
    if method == 'lw':
        result = bn.likelihood_weighting(node, evidence=evidence, N=10000)  # Specify N as a keyword argument
    elif method == 'gs':
        result = bn.gibbs_ask(node, evidence=evidence, N=10000)  # Specify N as a keyword argument
    elif method == 'mh':
        result = bn.metropolis_hastings(node, evidence=evidence, N=10000, p=p)  # Specify N and p as keyword arguments
        method += f'_{p}'  # Add the p value to the method name
    endtime = time.time()
    execution_time = endtime - starttime
    result_dict = {'Time': str(execution_time),'P_False': result[0],'P_True': result[1]}
    return node, method, result_dict

def main(): 
    # List of Bayesian Network files
    bn_files = ['10p.json', '10d.json', '25p.json', '25d.json', '50p.json', '50d.json', '100p.json', '100d.json']

    # List of p values
    p_values = [0.75, 0.85, 0.95]

    # Open the output file
    with open('experiment_evidence_results.txt', 'w') as f:
        # Loop over each Bayesian Network
        for bn_file in bn_files:
            f.write(f'Bayesian Network: {bn_file}\n')

            # Select nodes of interest
            nodes_of_interest = select_nodes_of_interest(bn_file)
            f.write(f'Nodes of interest: {nodes_of_interest}\n')

            # Evidence variables
            evidence_nodes = select_evidence_nodes(bn_file, nodes_of_interest) 

            # Define 1 or 0 for each evidence and use the same values throughout the experiment
            evidence_values = {evidence_node: np.random.choice([0, 1]) for evidence_node in evidence_nodes}

            # Loop over each evidence variable
            for evidence_node in evidence_nodes:
                # Print the evidence values
                f.write(f'Evidence: {{\'{evidence_node}\': {evidence_values[evidence_node]}}}\n')

                # Prepare a list of arguments for multiprocessing
                args_list = [(bn_file, node, method, None, {evidence_node: evidence_values[evidence_node]}) for node in nodes_of_interest for method in ['lw', 'gs']]
                args_list += [(bn_file, node, 'mh', p, {evidence_node: evidence_values[evidence_node]}) for node in nodes_of_interest for p in p_values]

                # Use a multiprocessing pool to run the methods in parallel
                with Pool() as pool:
                    method_results = pool.map(run_method, args_list)

                # Loop over each node of interest again to write the results to the file
                for node in nodes_of_interest:
                    f.write(f'Node: {node}\n')

                    # Write the exact inference result
                    exact_inference_result = run_exact_inference(bn_file, node)
                    f.write(f'Exact inference: {exact_inference_result}\n')

                    # Write the results of the methods
                    for node_res, method, result_dict in method_results:
                        if node_res == node:
                            f.write(f'{method}: {result_dict}\n')



if __name__ == "__main__":
    main()

