import json
import random
import argparse
import time
import numpy as np
import subprocess

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
                parents = self.bn_data[node]['parents']
                parent_values = [sample[str(parent)] for parent in parents if str(parent) in sample]
                probs = [prob_list[1] for prob_list in self.bn_data[node]['prob'] if prob_list[0] == parent_values]
                if probs:
                    sample[node] = 1 if random.random() < probs[0] else 0
                else:
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
        # Initialize all variables to random values
        X = {var: np.random.choice([0, 1]) for var in self.bn_data}
        X.update(evidence)

        counts = {0: 0, 1: 0}

        for _ in range(N):
            if np.random.random() < p:
                # Perform Gibbs sampling
                for Zi in X:
                    if Zi not in evidence:
                        # Sample a value for Zi given its Markov blanket
                        X[Zi] = self.sample_given_markov_blanket(Zi, X)
            else:
                # Generate a random sample
                X, _ = self.weighted_sample(X, p)

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
        node_importance[node] = num_parents + num_children

    # Sort the nodes by their importance (number of parents + number of children)
    sorted_nodes = sorted(node_importance, key=node_importance.get, reverse=True)

    # Select approximately 40% of the nodes
    num_nodes_of_interest = int(np.ceil(0.4 * len(sorted_nodes)))
    nodes_of_interest = sorted_nodes[:num_nodes_of_interest]

    return nodes_of_interest

def run_exact_inference(bn_file, query_var):
    # Run the exact inference script and capture its output
    result = subprocess.run(['python', 'exact-inference.py', '-f', bn_file, '-q', str(query_var)], capture_output=True, text=True)

    # Parse the output to extract the time and probabilities
    lines = result.stdout.split('\n')
    time_line = next(line for line in lines if line.startswith('Time'))
    time = float(time_line.split(':')[1].strip())

    p_false_line = next(line for line in lines if line.startswith('P_False'))
    p_false = float(p_false_line.split(':')[1].strip())

    p_true_line = next(line for line in lines if line.startswith('P_True'))
    p_true = float(p_true_line.split(':')[1].strip())

    return {'Time': time, 'P_False': p_false, 'P_True': p_true}

def main():
    parser = argparse.ArgumentParser(description="Bayesian Network Inference")
    parser.add_argument('-f', '--file', required=True, help='Path to the JSON file containing the Bayesian network structure.')
    parser.add_argument('-q', '--query', required=True, help='The query variable name.')
    parser.add_argument('-e', '--evidence', nargs='*', default=[], help='Evidence variables and their values (e.g., E1 1 E2 0).')
    parser.add_argument('-N', type=int, default=10000, help='Number of samples.')
    parser.add_argument('-p', type=float, default=0.5, help='Probability parameter (default: 0.5).')
    parser.add_argument('-m', '--method', required=True, help='The inference method to use. Acceptable values are "lw" for Likelihood Weighting, "gs" for Gibbs Sampling, and "mh" for Metropolis Hastings.')
    args = parser.parse_args()

    bn = BayesianNetwork(args.file)
    evidence = {args.evidence[i]: int(args.evidence[i + 1]) for i in range(0, len(args.evidence), 2)}

    print("Running {} on query variable {}".format(args.method, args.query))

    starttime = time.time()
    if args.method == 'lw':
        result = bn.likelihood_weighting(args.query, evidence, args.N)
    elif args.method == 'gs':
        result = bn.gibbs_ask(args.query, evidence, args.N)
    elif args.method == 'mh':
        result = bn.metropolis_hastings(args.query, evidence, args.N, args.p)
    else:
        raise ValueError("Invalid method. Acceptable values are 'lw', 'gs', and 'mh'.")
    endtime = time.time()
    execution_time = endtime - starttime

    for val, prob in result.items():
        print("Probability of {} is {}: {}".format(args.query, val, prob))

    print("-----Finished!-----")
    print("Time\t:", execution_time, "secs")
    print("P_False\t:", result[0])
    print("P_True\t:", result[1])

if __name__ == "__main__":
    main()
