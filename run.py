"""
Author: Duncan Kirby
Date: 2021-11-04
Run the following command before running any python files in this directory:
    source network/bin/activate
"""
import immune_network
import graph_evolution
import numpy as np
import os

if __name__ == "__main__":
    netPop = graph_evolution.NetworkPopulation()
    fit_hist = netPop.run(cpu=8)
    np.save(os.path.join(os.getcwd(), 'output', 'fit_hist.npy', fit_hist)
