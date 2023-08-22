import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import threading
import time

import torch
from joblib import load

from sls_ml.af_nn_model_creator import AAF_GCNConv
from sls_ml.af_parser import parse_file
from sls_ml.walkaaf import walkaaf_with_ml2, walkaaf, walkaaf_with_ml3, walkaaf_with_ml1, walkaaf_with_ml3_nn, \
    walkaaf_with_ml1_nn, walkaaf_with_ml2_nn


def evaluate_algorithm_for_g(af_graph, model_flip, model_in, g, num_runs=5):
    success_count = 0
    total_time = 0

    for i in range(num_runs):
        start_time = time.time()

        result = run_algorithm_with_timeout(af_graph, walkaaf_with_ml3_nn, model_flip, model_in, g)

        elapsed_time = time.time() - start_time

        if result is not None:
            total_time += elapsed_time
            success_count += 1
            print(f"Run {i} for g={g} success!")
        else:
            print(f"Run {i} for g={g} failed!")

    # Calculate average time and success rate
    avg_time = total_time / num_runs if success_count > 0 else 150  # If all runs time out, default to 10 mins
    success_rate = success_count / num_runs

    return avg_time, success_rate


def test_for_best_parameter(af_graphs, model_flip, model_in):
    g_values = np.arange(0.1, 1.1, 0.2)

    all_avg_times = []
    all_success_rates = []

    for g in g_values:
        total_time = 0
        total_success = 0
        for af_graph in af_graphs:
            avg_time, success_rate = evaluate_algorithm_for_g(af_graph, model_flip, model_in, g)
            total_time += avg_time
            total_success += success_rate

        all_avg_times.append(total_time / len(af_graphs))
        all_success_rates.append(total_success / len(af_graphs))

    return g_values, all_avg_times, all_success_rates


def run_algorithm_with_timeout(af_graph, algorithm, *args):
    result = [None]

    def worker():
        result[0] = algorithm(af_graph, *args)

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout=360)

    if thread.is_alive():
        print(f"Aborting due to timeout!")


    return result[0]

def test_algorithm(af_graphs, algorithm, *args):
    success_count = 0
    total_time = 0

    for af_graph in af_graphs:
        start_time = time.time()
        result = run_algorithm_with_timeout(af_graph, algorithm, *args)
        end_time = time.time()

        total_time += end_time - start_time

        if result is not None:
            print('success')
            success_count += 1
        else:
            print('no')

    average_time = total_time / len(af_graphs)
    success_rate = success_count / len(af_graphs)

    print(success_count)
    return average_time, success_rate


def evaluate_walkaaf(avg_time_walkaaf, avg_time_ml, success_rate_walkaaf, success_rate_ml):
    algorithms = ['Vanilla WalkAAF', 'WalkAAF with ML']
    avg_times = [avg_time_walkaaf, avg_time_ml]
    success_rates = [success_rate_walkaaf, success_rate_ml]

    # Setting the bar width
    bar_width = 0.35
    r1 = np.arange(len(avg_times))
    r2 = [x + bar_width for x in r1]

    plt.figure(figsize=(10, 6))


    plt.bar(r1, avg_times, color='blue', width=bar_width, edgecolor='grey', label='Avg Time')

    plt.bar(r2, success_rates, color='red', width=bar_width, edgecolor='grey', label='Success Rate')

    plt.title('Comparison between WalkAAF algorithms', fontweight='bold')
    plt.xlabel('Algorithms', fontweight='bold')

    plt.xticks([r + bar_width for r in range(len(avg_times))], algorithms)

    plt.legend()
    plt.show()




def load_af_graphs_from_directory(path):
    af_graphs_list = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        af_graph = parse_file(filepath)
        af_graphs_list.append(af_graph)
    return af_graphs_list

def save_plot(filename):
    plt.savefig(filename, bbox_inches='tight')
    plt.close()



def experiments():
    model_in_red = AAF_GCNConv(4, 2)

    output_folder = "/Users/konrad_bsc/Documents/GitHub/sls-ml/files/ml_models"
    PATH = os.path.join(output_folder, "gcn_nn_in_red.pt")
    model_in_red.load_state_dict(torch.load(PATH))
    model_in_red.eval()

    model_rn_red = AAF_GCNConv(5, 2)
    PATH = os.path.join(output_folder, "gcn_nn_rn_red.pt")
    model_rn_red.load_state_dict(torch.load(PATH))
    model_rn_red.eval()

    path = '/Users/konrad_bsc/Documents/GitHub/sls-ml/files/benchmark_aaf'
    af_graphs_list = load_af_graphs_from_directory(path)

    model_rn = load(
        '/Users/konrad_bsc/Documents/GitHub/sls-ml/files/ml_models/trained_model_RandomForest_rn_red.joblib')
    model_in = load(
        '/Users/konrad_bsc/Documents/GitHub/sls-ml/files/ml_models/trained_model_RandomForest_in_red.joblib')

    algorithms = [
       # {'function': walkaaf, 'name': 'walkaaf'},
       # {'function': walkaaf_with_ml1, 'args': [model_rn], 'name': 'walkaaf_with_ml1'},
       # {'function': walkaaf_with_ml2, 'args': [model_in], 'name': 'walkaaf_with_ml2'},
      #  {'function': walkaaf_with_ml3, 'args': [model_rn, model_in], 'name': 'walkaaf_with_ml3'},
        {'function': walkaaf_with_ml1_nn, 'args': [model_rn_red], 'name': 'walkaaf_with_ml1_nn'},
        {'function': walkaaf_with_ml2_nn, 'args': [model_in_red], 'name': 'walkaaf_with_ml2_nn'},
        {'function': walkaaf_with_ml3_nn, 'args': [model_rn_red, model_in_red], 'name': 'walkaaf_with_ml3_nn'},
    ]

    results = []

    for algo in algorithms:
        print(f"Testing {algo['name']}...")
        avg_time, success_rate = test_algorithm(af_graphs_list, algo['function'], *algo.get('args', []))
        results.append({
            'Algorithm': algo['name'],
            'Avg Time': avg_time,
            'Success Rate': success_rate
        })

    with open('evaluation_results_nn.csv', 'w', newline='') as csvfile:
        fieldnames = ['Algorithm', 'Avg Time', 'Success Rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Evaluation completed. Results saved to evaluation_results.csv.")

if __name__ == '__main__':
    model_in_red = AAF_GCNConv(4, 2)

    output_folder = "/Users/konrad_bsc/Documents/GitHub/sls-ml/files/ml_models"
    PATH = os.path.join(output_folder, "gcn_nn_in_red.pt")
    model_in_red.load_state_dict(torch.load(PATH))
    model_in_red.eval()

    model_rn_red = AAF_GCNConv(5, 2)
    PATH = os.path.join(output_folder, "gcn_nn_rn_red.pt")
    model_rn_red.load_state_dict(torch.load(PATH))
    model_rn_red.eval()

    path = '/Users/konrad_bsc/Documents/GitHub/sls-ml/files/benchmark_aaf'
    af_graphs_list = load_af_graphs_from_directory(path)

    model_rn = load('/Users/konrad_bsc/Documents/GitHub/sls-ml/files/ml_models/trained_model_RandomForest_rn_red.joblib')
    model_in = load(
        '/Users/konrad_bsc/Documents/GitHub/sls-ml/files/ml_models/trained_model_RandomForest_in_red.joblib')


    experiments()
    #avg_time_walkaaf, success_rate_walkaaf = test_algorithm(af_graphs_list, walkaaf)
    #print(f"Vanilla WalkAAF - Avg Time: {avg_time_walkaaf}, Success Rate: {success_rate_walkaaf}")

    # Testing walkaaf_with_ml
    # avg_time_ml, success_rate_ml = test_algorithm(af_graphs_list, walkaaf_with_ml1, model_rn)
    # print(f"WalkAAF with ML - Avg Time: {avg_time_ml}, Success Rate: {success_rate_ml}")


    # Testing walkaaf_with_ml
    # avg_time_ml, success_rate_ml = test_algorithm(af_graphs_list, walkaaf_with_ml2, model_in)
    # print(f"WalkAAF with ML - Avg Time: {avg_time_ml}, Success Rate: {success_rate_ml}")


    # Testing walkaaf_with_ml
    # avg_time_ml, success_rate_ml = test_algorithm(af_graphs_list, walkaaf_with_ml3, model_rn, model_in)
    # print(f"WalkAAF with ML - Avg Time: {avg_time_ml}, Success Rate: {success_rate_ml}")

    # Visualize and save
     # evaluate_walkaaf(avg_time_walkaaf, avg_time_ml, success_rate_walkaaf, success_rate_ml)
    # save_plot('evaluation_comparison.png')





#    selected_graphs = random.sample(af_graphs_list, 15)

 #   g_values, avg_times, success_rates = test_for_best_parameter(selected_graphs, model_rn_red, model_in_red)

  #  # Visualization
   # plt.figure(figsize=(12, 6))

    #plt.subplot(1, 2, 1)
    #plt.plot(g_values, avg_times, marker='o')
    #plt.xlabel('g value')
    #plt.ylabel('Average Time (s)')

    #plt.subplot(1, 2, 2)
    #plt.plot(g_values, success_rates, marker='o', color='green')
    #plt.xlabel('g value')
    #plt.ylabel('Success Rate')

    plt.tight_layout()
    save_plot('average_evaluation_comparison.png')
