#!/usr/local/bin/python3
# encoding: utf-8

import sys

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from argparse import ArgumentParser
from ziwm.tools import feature_extractor

def main(argv=None):

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    # Setup argument parser
    parser = ArgumentParser(description='Preprocess data into dataset with extracted features')
    parser.add_argument("data_filename", help="Text file with data as matrix. \
        Should contain header with classes of columns - 'c' for category, 'r' for real and 'y' for output")
    parser.add_argument("output_filename", help="Filename under which output dataset will be saved")

    # Process arguments
    args = parser.parse_args()

    data_filename = args.data_filename
    output_filename = args.output_filename
    
    raw_data, column_types = feature_extractor.load_file(data_filename)
    
    X_data, X_feature_indices = feature_extractor.create_features(raw_data, column_types)
    
    feature_extractor.save_to_file(output_filename, data=X_data, header=X_feature_indices)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())