#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Libraries
from src import main
import yaml
import os
import argparse

parser = argparse.ArgumentParser(description = 'Extraction of the characteristics')
parser.add_argument('--config', default = "config.yml", help = "Name of (including the path to) the configuration file", type=str)
args = parser.parse_args()

def run():

    # open the configuration file
    config = args.config
    with open(os.path.join(config), "rb") as f:
        cf = yaml.load(f, Loader = yaml.FullLoader)

    # initialize the characteristics extraction module
    extraction = main.MetadataExtraction(cf = cf)

    # run it over the complete set of polygons.
    extraction.extract_all_characteristics()

# executable
if __name__ == '__main__':

    # Run the pipeline.
    run()

