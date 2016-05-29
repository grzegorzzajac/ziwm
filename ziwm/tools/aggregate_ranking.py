#!/usr/bin/python2.7

import os
import sys
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))


def load_data(dir_path):
    results_data = np.asarray([])
    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            file_data = np.genfromtxt(path.join(dir_path, file), dtype='str', delimiter=',', invalid_raise=False)
            if results_data.size == 0:
                results_data = file_data
            else:
                raw_data = file_data[1:]
                assert raw_data.shape[1] == results_data.shape[1]
                results_data = np.concatenate((results_data, raw_data))

    return results_data


def calculate_rank(results_data):
    # header = results_data[0]

    sizes = np.unique(results_data[1:, 7]).astype(int)
    sizes.sort()
    datasets = np.unique(results_data[1:, 1])
    rank_data = {}

    for size in sizes:
        rank_data[size] = {}
        for dataset in datasets:
            # Get results and sort
            rank_list = np.asarray([row for row in results_data if row[1] == dataset and row[7].astype(int) == size])
            if rank_list.size == 0:
                continue
            rank_list = rank_list[np.ix_(rank_list[:, 0].argsort()[::-1], [5, 6, 8])]
            rank_list = [tuple(row) for row in rank_list]

            for entry in rank_list:
                rank = rank_list.index(entry)
                if entry in rank_data[size].keys():
                    rank_data[size][entry] += rank
                else:
                    rank_data[size][entry] = rank

    # 0     1       2             3            4                 5     6        7             8
    # score,dataset,feature_count,dataset_size,number_of_classes,model,ensemble,ensemble_size,voting_system

    array_data = []
    for size in rank_data.keys():
        items = []
        for k in rank_data[size].keys():
            row = [size] + list(k) + [rank_data[size][k]]
            items.append(row)

        items = [item + [i+1] for i, item in enumerate(sorted(items, key=lambda x: x[4]))]
        array_data += items
    return array_data, "ensemble_size,model,ensemble,voting_system,aggregated_rank,total_rank"


def main(argv=None):

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    # Setup argument parser
    parser = ArgumentParser(description='Aggregate individual scores into a global rank')
    parser.add_argument("results_path", help="Path to directory with results data.")
    parser.add_argument("output_filename", help="Filename under which output ranking data will be saved")

    # Process arguments
    args = parser.parse_args()

    results_path = args.results_path
    output_filename = args.output_filename

    # Load data and calculate ranks
    results_data = load_data(results_path)

    rank_data, header = calculate_rank(results_data)

    # Save the data
    np.savetxt(output_filename, rank_data, header=header, fmt='%s', delimiter=',')

    return 0

if __name__ == "__main__":
    sys.exit(main())
