import os

import matplotlib.pyplot as plt
import numpy as np
import threading
import time

from joblib import load

from sls_ml.af_parser import parse_file
from sls_ml.walkaaf import walkaaf_with_ml2, walkaaf, walkaaf_with_ml3, walkaaf_with_ml1


def evaluate_algorithm_for_g(af_graph, model, g, num_runs=10):
    success_count = 0
    total_time = 0

    for i in range(num_runs):
        print(i)
        start_time = time.time()
        result = walkaaf_with_ml2(af_graph, model, g=g)
        total_time += time.time() - start_time

        if result is not None:
            success_count += 1

    avg_time = total_time / num_runs
    success_rate = success_count / num_runs
    print(success_count)
    return avg_time, success_rate


def test_for_best_parameter(af_graph, model):

    g_values = np.arange(0.1, 1.1, 0.1)

    avg_times = []
    success_rates = []

    for g in g_values:
        avg_time, success_rate = evaluate_algorithm_for_g(af_graph, model, g)
        avg_times.append(avg_time)
        success_rates.append(success_rate)

    # Visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(g_values, avg_times, marker='o')

    plt.xlabel('g value')
    plt.ylabel('Average Time (s)')

    plt.subplot(1, 2, 2)
    plt.plot(g_values, success_rates, marker='o', color='green')

    plt.xlabel('g value')
    plt.ylabel('Success Rate')

    plt.tight_layout()
    plt.show()


def run_algorithm_with_timeout(af_graph, algorithm, *args):
    result = [None]

    def worker():
        result[0] = algorithm(af_graph, *args)

    thread = threading.Thread(target=worker)
    thread.daemon = True  # Set the thread as a daemon
    thread.start()
    thread.join(timeout=150)  # 150 seconds

    if thread.is_alive():
        print("Aborting due to timeout!")
        # Optionally, don't wait for the thread to finish since it's a daemon now
        # thread.join()

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

    # Create bars for avg times
    plt.bar(r1, avg_times, color='blue', width=bar_width, edgecolor='grey', label='Avg Time')
    # Create bars for success rates
    plt.bar(r2, success_rates, color='red', width=bar_width, edgecolor='grey', label='Success Rate')

    # Title & Subtitle
    plt.title('Comparison between WalkAAF algorithms', fontweight='bold')
    plt.xlabel('Algorithms', fontweight='bold')

    # X axis
    plt.xticks([r + bar_width for r in range(len(avg_times))], algorithms)

    # Show the legend
    plt.legend()

    # Display the graph
    plt.show()




def load_af_graphs_from_directory(path):
    af_graphs_list = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        # Using your parse_file function to load the AAF graphs
        af_graph = parse_file(filepath)
        af_graphs_list.append(af_graph)
    return af_graphs_list

def save_plot(filename):
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':



    path = '/Users/konraddrees/Documents/GitHub/sls-ml/files/benchmark_aaf'
    af_graphs_list = load_af_graphs_from_directory(path)

    model_rn = load('/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models/trained_model_RandomForest_rn_red.joblib')
    model_in = load(
        '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_models/trained_model_RandomForest_in_red.joblib')

    avg_time_walkaaf, success_rate_walkaaf = test_algorithm(af_graphs_list, walkaaf)
    print(f"Vanilla WalkAAF - Avg Time: {avg_time_walkaaf}, Success Rate: {success_rate_walkaaf}")

    # Testing walkaaf_with_ml
    # avg_time_ml, success_rate_ml = test_algorithm(af_graphs_list, walkaaf_with_ml1, model_rn)
    # print(f"WalkAAF with ML - Avg Time: {avg_time_ml}, Success Rate: {success_rate_ml}")


    # Testing walkaaf_with_ml
    # avg_time_ml, success_rate_ml = test_algorithm(af_graphs_list, walkaaf_with_ml2, model_in)
    # print(f"WalkAAF with ML - Avg Time: {avg_time_ml}, Success Rate: {success_rate_ml}")


    # Testing walkaaf_with_ml
    avg_time_ml, success_rate_ml = test_algorithm(af_graphs_list, walkaaf_with_ml3, model_rn, model_in)
    print(f"WalkAAF with ML - Avg Time: {avg_time_ml}, Success Rate: {success_rate_ml}")

    # Visualize and save
    evaluate_walkaaf(avg_time_walkaaf, avg_time_ml, success_rate_walkaaf, success_rate_ml)
    save_plot('evaluation_comparison.png')