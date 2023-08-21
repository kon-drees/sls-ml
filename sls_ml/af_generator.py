import math
import os
import subprocess
import matplotlib.pyplot as plt

from sls_ml.af_parser import parse_file


def generate_af_benchgen(num_args, af_type, output_folder, additional_parameters=""):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # location of the jar file
    command = f"java -jar /Users/konraddrees/Documents/AFBenchGen2/jAFBenchGen.jar -numargs {num_args} -type {af_type} {additional_parameters}"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    af = output.decode('utf-8')

    counter = 1
    af_name = f'af_{num_args}_{af_type}_{counter}.apx'
    while os.path.exists(os.path.join(output_folder, af_name)):
        counter += 1
        af_name = f'af_{num_args}_{af_type}_{counter}.apx'

    af_path = os.path.join(output_folder, af_name)
    with open(af_path, 'w') as f:
        f.write(af)
    return af_path


def generate_af():
    output_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/generated_argumentation_frameworks'
    af_types = ['ErdosRenyi', 'WattsStrogatz', 'BarabasiAlbert']
    num_args_list = list(range(5, 26, 1))


    ER_probAttacks_range = [0.2, 0.4, 0.6, 0.8]
    # WS_baseDegree_range = [6, 10, 16, 20]
    WS_beta_range = [0.5]
    BA_WS_probCycles_range = [0.2, 0.4, 0.6, 0.8]
    # using afbenchgen2
    for af_type in af_types:
        for num_args in num_args_list:
            for i in range(1, 31):
                additional_parameters = ""
                if af_type == 'ErdosRenyi':
                    for ER_probAttacks in ER_probAttacks_range:
                        additional_parameters = f"-ER_probAttacks {ER_probAttacks}"
                        generate_af_benchgen(num_args, af_type, output_folder, additional_parameters)
                        print(f"Generated AF with {num_args} Arguments, {af_type} structure.")

                elif af_type == 'WattsStrogatz':
                    for WS_beta in WS_beta_range:
                        for BA_WS_probCycles in BA_WS_probCycles_range:
                            min_base_degree = max(1, int(math.log(num_args,
                                                                  2)))  # Minimum base degree satisfying the condition
                            if min_base_degree % 2 != 0:  # Ensure minimum base degree is even
                                min_base_degree += 1
                            max_base_degree = num_args - 1  # Maximum base degree
                            WS_baseDegree = min_base_degree
                            if WS_baseDegree % 2 == 0 & WS_baseDegree < num_args:
                                additional_parameters = f"-WS_baseDegree {WS_baseDegree} -WS_beta {WS_beta} -BA_WS_probCycles {BA_WS_probCycles}"
                                generate_af_benchgen(num_args, af_type, output_folder, additional_parameters)
                                print(f"Generated AF with {num_args} Arguments, {af_type} structure.")
                            else:
                                print(f"Can't generate with Num Attacks {num_args} and degree {WS_baseDegree}!")

                elif af_type == 'BarabasiAlbert':
                    for BA_WS_probCycles in BA_WS_probCycles_range:
                        additional_parameters = f"-BA_WS_probCycles {BA_WS_probCycles}"
                        generate_af_benchgen(num_args, af_type, output_folder, additional_parameters)
                        print(f"Generated AF with {num_args} Arguments, {af_type} structure.")



def plot_node_distribution(directory_path):
    num_nodes = []
    total_af = 0
    total_nodes = 0
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        try:
            aaf_graph = parse_file(file_path)
            nodes = aaf_graph.number_of_nodes()
            num_nodes.append(nodes)
            total_nodes += nodes
            total_af += 1
        except ValueError as ve:
            print(f"Error parsing file {file_path}: {ve}")
            continue

    plt.hist(num_nodes, bins=30, edgecolor='black')
    plt.title("Distribution of Number of Nodes in Argumentation Frameworks")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Number of Frameworks")
    plt.show()

    print(f"Total number of Argumentation Frameworks: {total_af}")
    print(f"Total number of Arguments across all Argumentation Frameworks: {total_nodes}")


if __name__ == '__main__':
   generate_af()
   directory_path = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'
   plot_node_distribution(directory_path)

