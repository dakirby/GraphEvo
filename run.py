"""
Author: Duncan Kirby
Date: 2021-11-04
Run the following command before running any python files in this directory:
    source network/bin/activate
"""
import immune_network
import graph_evolution
from networkx.drawing.layout import shell_layout
import numpy as np
import os

if __name__ == "__main__":
    netPop = graph_evolution.NetworkPopulation()
    fit_hist = netPop.run()
