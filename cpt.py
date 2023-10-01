def read_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    return lines

def parse_data(lines):
    results = {}
    current_bn = None
    current_node = None
    for line in lines:
        if "Bayesian Network" in line:
            current_bn = line.split(':')[1].strip().split('.json')[0]
            results[current_bn] = {}
        elif "Node" in line:
            current_node = line.split(':')[1].strip()
            if current_node not in results[current_bn]:
                results[current_bn][current_node] = {}
        elif any(method in line for method in ["Exact inference", "lw", "gs", "mh_0.75", "mh_0.85", "mh_0.95"]):
            method = line.split(':')[0].strip()
            time = float(line.split("'Time': '")[1].split("'")[0])
            p_false = float(line.split("'P_False': ")[1].split(",")[0])
            p_true = float(line.split("'P_True': ")[1].split("}")[0])
            results[current_bn][current_node][method] = {'Time': time, 'P_False': p_false, 'P_True': p_true}
    return results

def calculate_relative_effectiveness(results):
    output = ""
    for bn_name, bn_data in results.items():
        output += f'Bayesian Network: {bn_name}\n'
        methods = ["Exact inference", "lw", "gs", "mh_0.75", "mh_0.85", "mh_0.95"]
        averages = {method: 0 for method in methods}
        for node, node_data in bn_data.items():
            if 'Exact inference' in node_data:
                exact_inference = node_data['Exact inference']
                for method, method_data in node_data.items():
                    if method != 'Exact inference':
                        relative_effectiveness = abs(exact_inference['P_True'] - method_data['P_True'])
                        averages[method] += relative_effectiveness
        num_nodes = len(bn_data)
        for method in methods:
            if method != 'Exact inference':
                averages[method] /= num_nodes
                output += f'Method: {method}, Average Relative Effectiveness: {averages[method]}\n'
    return output

def write_to_file(output, file_name):
    with open(file_name, 'w') as file:
        file.write(output)

lines = read_file('experiment_evidence_results.txt')
results = parse_data(lines)
output = calculate_relative_effectiveness(results)
write_to_file(output, 'cpt.txt')
