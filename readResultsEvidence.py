import json
import numpy as np

def read_experiment_results(file_name): 
    results = {}
    current_bn = None
    current_node = None
    current_evidence = None

    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            components = line.strip().split(': ', 1)
            if len(components) == 2:
                key, value = components
                if key == 'Bayesian Network':
                    current_bn = value.replace("'", "")
                elif key == 'Nodes of interest':
                    continue
                elif key == 'Evidence':
                    current_evidence = json.loads(value.replace("'", "\""))
                elif key == 'Node':
                    current_node = value.replace("'", "")
                elif key in ['Exact inference', 'lw', 'gs', 'mh_0.75', 'mh_0.85', 'mh_0.95']:
                    value_dict = json.loads(value.replace("'", "\""))
                    results[(current_bn, current_node, key, str(current_evidence))] = value_dict

    return results

def analyze_results(results):
    run_times = {}
    p_diff_values = {}

    for (bn, node, method, evidence), value in results.items():
        if (bn, method, evidence) not in run_times:
            run_times[(bn, method, evidence)] = []
        if (bn, method, evidence) not in p_diff_values and method != 'Exact inference':
            p_diff_values[(bn, method, evidence)] = []

        run_times[(bn, method, evidence)].append(float(value['Time']))
        if method != 'Exact inference':
            exact_inference_result = results[(bn, node, 'Exact inference', evidence)]
            p_diff = abs(value['P_False'] - exact_inference_result['P_False']) + abs(value['P_True'] - exact_inference_result['P_True'])
            p_diff_values[(bn, method, evidence)].append(p_diff)

    return run_times, p_diff_values

def write_analysis_results(run_times, p_diff_values):
    with open('read_evidence_results.txt', 'w') as f:
        for (bn, method, evidence) in run_times.keys():
            if run_times[(bn, method, evidence)]:  # Check if there are run times
                mean_run_time = np.mean(run_times[(bn, method, evidence)])
                std_dev_run_time = np.std(run_times[(bn, method, evidence)])
                f.write(f'Bayesian Network: {bn}, Method: {method}, Evidence: {evidence}\n')
                f.write(f'Mean run time: {mean_run_time}\n')
                f.write(f'Standard deviation of run time: {std_dev_run_time}\n')
            if method != 'Exact inference' and p_diff_values[(bn, method, evidence)]:  # Check if there are probability differences
                variation_in_converged_probabilities = np.var(p_diff_values[(bn, method, evidence)])
                f.write(f'Variation in converged probabilities: {variation_in_converged_probabilities}\n')
            f.write('\n')
        f.write('\n')

def main(): 
    results = read_experiment_results('experiment_evidence_results.txt')
    run_times, p_diff_values = analyze_results(results)
    write_analysis_results(run_times, p_diff_values)

if __name__ == '__main__':
    main()
