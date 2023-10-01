import json
import numpy as np
import matplotlib.pyplot as plt

def read_experiment_results(file_name): 
    results = {}
    current_bn = None
    current_node = None

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
                elif key == 'Node':
                    current_node = value.replace("'", "")
                elif key in ['Exact inference', 'lw', 'gs', 'mh_0.75', 'mh_0.85', 'mh_0.95']:
                    value_dict = json.loads(value.replace("'", "\""))
                    results[(current_bn, current_node, key)] = value_dict

    return results

def analyze_results(results):
    run_times = {bn: {method: [] for method in ['Exact inference', 'lw', 'gs', 'mh_0.75', 'mh_0.85', 'mh_0.95']} for bn in set(bn for bn, _, _ in results.keys())}
    p_diff_values = {bn: {method: [] for method in ['lw', 'gs', 'mh_0.75', 'mh_0.85', 'mh_0.95']} for bn in set(bn for bn, _, _ in results.keys())}

    for (bn, node, method), value in results.items():
        run_times[bn][method].append(float(value['Time']))
        if method != 'Exact inference':
            exact_inference_result = results[(bn, node, 'Exact inference')]
            p_diff = abs(value['P_False'] - exact_inference_result['P_False']) + abs(value['P_True'] - exact_inference_result['P_True'])
            p_diff_values[bn][method].append(p_diff)

    return run_times, p_diff_values

def print_analysis_results(run_times, p_diff_values):
    for bn in run_times.keys():
        print(f'Bayesian Network: {bn}')
        for method in run_times[bn].keys():
            mean_run_time = np.mean(run_times[bn][method])
            std_dev_run_time = np.std(run_times[bn][method])
            print(f'Method: {method}')
            print(f'Mean run time: {mean_run_time}')
            print(f'Standard deviation of run time: {std_dev_run_time}')
            if method != 'Exact inference':
                variation_in_converged_probabilities = np.var(p_diff_values[bn][method])
                print(f'Variation in converged probabilities: {variation_in_converged_probabilities}')
            print()
        print()

def write_analysis_results(run_times, p_diff_values):
    with open('read_results.txt', 'w') as f:
        for bn in run_times.keys():
            f.write(f'Bayesian Network: {bn}\n')
            for method in run_times[bn].keys():
                mean_run_time = np.mean(run_times[bn][method])
                std_dev_run_time = np.std(run_times[bn][method])
                f.write(f'Method: {method}\n')
                f.write(f'Mean run time: {mean_run_time}\n')
                f.write(f'Standard deviation of run time: {std_dev_run_time}\n')
                if method != 'Exact inference':
                    variation_in_converged_probabilities = np.var(p_diff_values[bn][method])
                    f.write(f'Variation in converged probabilities: {variation_in_converged_probabilities}\n')
                f.write('\n')
            f.write('\n')

def main(): 
    results = read_experiment_results('experiment_results.txt')
    run_times, p_diff_values = analyze_results(results)
    write_analysis_results(run_times, p_diff_values)

if __name__ == '__main__':
    main()
