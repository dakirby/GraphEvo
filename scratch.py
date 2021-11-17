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


testNet = immune_network.ImmuneGraph()
testNet.layout = shell_layout

testNet.new_interaction()
testNet.delete_interaction()
testNet.mutate_interaction()
testNet.duplicate_protein()
testNet.delete_protein()

y0Test = testNet.equilibrate()
#testNet.draw_graph()
# t = (0., 20.)
# testNet.plot_simulation(t)
# _, score = testNet.run_graph(t, y0=y0Test, score=True)
# print(score)

ne = graph_evolution.NetworkPopulation(num_generations=2, num_individuals=20)
fit_hist = ne.run()
print(fit_hist)

# ne = graph_evolution.NetworkPopulation()
# ne.load(os.path.join(os.getcwd(), 'output', 'g19', 'Pop.pkl'))
# print(ne.num_individuals)
# ne.individuals[0].immune_network.draw_graph()
