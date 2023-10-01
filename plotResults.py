import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to read the data from the file without evidence
def read_data_without_evidence(file_name):
    data = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Bayesian Network" in line:
                bn = line.split(": ")[1].strip()
            elif "Method" in line:
                method = line.split(": ")[1].strip()
                evidence = None  # No evidence
            elif "Mean run time" in line:
                mean_rt = float(line.split(": ")[1].strip())
            elif "Standard deviation of run time" in line:
                sd_rt = float(line.split(": ")[1].strip())
            elif "Variation in converged probabilities" in line:
                var_p = float(line.split(": ")[1].strip())
                data.append([bn, method, evidence, mean_rt, sd_rt, var_p])
    return pd.DataFrame(data, columns=['bn', 'method', 'evidence', 'mean_rt', 'sd_rt', 'var_p'])

# Function to read the data from the file with evidence
def read_data_with_evidence(file_name):
    data = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Bayesian Network" in line:
                bn, method, evidence = line.split(", ")
                bn = bn.split(": ")[1].strip()
                method = method.split(": ")[1].strip()
                evidence = evidence.split(": ")[1].strip()
            elif "Mean run time" in line:
                mean_rt = float(line.split(": ")[1].strip())
            elif "Standard deviation of run time" in line:
                sd_rt = float(line.split(": ")[1].strip())
            elif "Variation in converged probabilities" in line:
                var_p = float(line.split(": ")[1].strip())
                data.append([bn, method, evidence, mean_rt, sd_rt, var_p])
    return pd.DataFrame(data, columns=['bn', 'method', 'evidence', 'mean_rt', 'sd_rt', 'var_p'])

# Read the data
data_without_evidence = read_data_without_evidence("read_results.txt")
data_with_evidence = read_data_with_evidence("read_evidence_results.txt")

# Combine the data
data = pd.concat([data_without_evidence, data_with_evidence])

# Function to create the plots
def create_plots(data):
    methods = data['method'].unique()
    bns = data['bn'].unique()
    for method in methods:
        fig, axs = plt.subplots(1, len(bns), figsize=(15, 10))
        axs = np.array(axs)  # Ensure axs is an array
        for j, bn in enumerate(bns):
            df = data[(data['method'] == method) & (data['bn'] == bn)]
            # Adjust the indexing based on the number of bns
            ax = axs[j] if len(bns) > 1 else axs
            ax.boxplot([df['mean_rt'], df['sd_rt'], df['var_p']], labels=['Mean RT', 'SD RT', 'Var P'])
            ax.set_title(f'{method} - {bn}')
        plt.suptitle(f'{method} Comparison')
        plt.tight_layout()
        plt.savefig(f'{method}.png')
# Create the plots
create_plots(data)
