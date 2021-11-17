import immune_network
import graph_evolution
from networkx.drawing.layout import shell_layout
import numpy as np
import os
import unittest
import matplotlib.pyplot as plt
import copy

np.random.seed(seed=42)
global_ImmuneGraph = immune_network.ImmuneGraph()


class TestImmuneGraphMethods(unittest.TestCase):

    def test_init(self):
        """
        Ensures that the graph initiation process isnt' changed without
        throwing a failed unit test.
        """
        graph_dict = {}
        true_graph_dict = {('D', 'c0'): 1.6433127155860012, ('c0', 'r0'): 0.8708684897640193, ('r0', 't0'): 1.9111182426600926, ('t0', 'E'): 4.586099388741997}
        for ed in global_ImmuneGraph.graph.edges():
            w = global_ImmuneGraph.graph.get_edge_data(*ed)['weight']
            graph_dict.update({ed: w})
        self.assertEqual(graph_dict, true_graph_dict)

    def test_run(self):
        """
        Tests:
        1) Equilibration agrees with solution from Mathematica
            (see graph_solutions.nb)
        2) ODE integration yields same results as Mathematica ODE integrator
        3) Scoring via numerical integration yields same result too
        """

        # Test 1
        t = (0., 20.)
        y0 = global_ImmuneGraph.equilibrate()
        cyt_eq = y0[2] - 24.6838  # cytokine equilibrium
        rec_eq = y0[3] - 0.03412  # receptor equilibrium
        tfr_eq = y0[5] - 6.7E-24  # transcription factor equilibrium
        eff_eq = y0[-1] - 1.0000  # effector cell equilibrium
        y0Test = np.array([cyt_eq, rec_eq, tfr_eq, eff_eq])
        self.assertTrue(np.all(y0Test < 0.0001))

        # Test 2
        y0[0] = 0.5  # set parasite to 0.5
        y0[1] = 0.01  # set detector cell to 0.01
        y0[-1] = 0.01  # set effector cell to 0.01
        sim, score = global_ImmuneGraph.run_graph(t, y0=y0, score=True)
        final_vec = sim.y[:,-1]
        Math_final_vec = [0.00421984, 0.08538, 26.8394, 0.0314334, 0.968567, -9.55605E-38, 1., 1.]
        rel_err = np.subtract(Math_final_vec, final_vec)
        self.assertTrue(np.all(rel_err < 0.01))

        # Test 3
        mathScore = 0.325893  # From computation in Mathematica
        self.assertTrue(score[0]-mathScore < 0.001)


class TestNetworkEvolverMethods(unittest.TestCase):

    def test_evolve_graph(self):
        """
        Tests each of the graph mutation functions:
            - duplicate_protein()
            - new_interaction()
            - mutate_interaction()
            - delete_interaction()
            - delete_protein()
        """
        ne = graph_evolution.NetworkEvolver()
        ne.immune_network = global_ImmuneGraph
        # -------------------- #
        # duplicate_protein()
        # -------------------- #
        # sets the rng to generate a call that results in duplicating TF node
        for _ in range(8):
            np.random.random()
        ne.evolve_graph(DEBUG=True, DEBUG_decision=0.85)  # duplicate
        wE0 = ne.immune_network.graph.get_edge_data('t0', 'E')
        wr0 = ne.immune_network.graph.get_edge_data('t0', 'r0')
        wE1 = ne.immune_network.graph.get_edge_data('t1', 'E')
        wr1 = ne.immune_network.graph.get_edge_data('t1', 'r0')
        self.assertEqual(wE0, wE1)
        self.assertEqual(wr0, wr1)

        # -------------------- #
        # new_interaction()
        # -------------------- #
        ne.evolve_graph(DEBUG=True, DEBUG_decision=0.1)  # new interaction
        new_int = ne.immune_network.graph.get_edge_data('t1', 't0')
        self.assertEqual({'weight': -0.7033438017074073}, new_int)

        # -------------------- #
        # mutate_interaction()
        # -------------------- #
        for _ in range(3):  # set rng to the (t0, t1) interaction
            np.random.random()
        ne.evolve_graph(DEBUG=True, DEBUG_decision=0.7)  # mutate interaction
        wNew = ne.immune_network.graph.get_edge_data('t1', 't0')
        self.assertEqual(wNew, {'weight': -0.47160898629074743})

        # -------------------- #
        # delete_interaction()
        # -------------------- #
        self.assertTrue(('E', 't1') in ne.immune_network.graph.edges())
        ne.evolve_graph(DEBUG=True, DEBUG_decision=0.3)  # delete interaction
        self.assertFalse(('E', 't1') in ne.immune_network.graph.edges())

        # -------------------- #
        # delete_protein()
        # -------------------- #
        for _ in range(7):  # set rng so that we end up deleting the t1 protein
            np.random.random()
        ne.evolve_graph(DEBUG=True, DEBUG_decision=0.95)  # delete protein
        self.assertFalse(('r0', 't1') in ne.immune_network.graph.edges())
        self.assertFalse(('t0', 't1') in ne.immune_network.graph.edges())


class TestNetworkPopulationMethods(unittest.TestCase):
    """
    Tests the run() method to:
    1) ensure that each graph evolves independently
    2) selection per generation happens according to relative fitness
    3) multiprocessing correctly repeats evolution behaviour
    """

    def test_independent_and_selection(self):
        netPop = graph_evolution.NetworkPopulation(num_generations=1, num_individuals=4)

        # ------------------ #
        # Testing (1)
        # ------------------ #
        for i in range(4):
            netPop.individuals[i].immune_network.duplicate_protein()
        netPop.individuals[1].immune_network.delete_interaction()
        edges_before = []
        score_before = []
        for i in netPop.individuals:
            edges_before.append(i.immune_network.graph.edges())
            y0 = i.immune_network.equilibrate()
            y0[0] = 0.5  # initialize parasite to 0.5 for scoring run
            y0[1] = 0.01  # initialize detecotr cell to 0.01 for scoring run
            y0[-1] = 0.01  # initialize effector cell to 0.01 for scoring run
            t = (0, 20.)
            _, score = i.immune_network.run_graph(t, y0=y0, score=True)
            score_before.append(score[0])

        popfit, norm_rel_fit = netPop.run(DEBUG=True)
        edges_after = [i.immune_network.graph.edges() for i in netPop.individuals]
        score_after = []
        for i in netPop.individuals:
            edges_before.append(i.immune_network.graph.edges())
            y0 = i.immune_network.equilibrate()
            y0[0] = 0.5  # initialize parasite to 0.5 for scoring run
            y0[1] = 0.01  # initialize detecotr cell to 0.01 for scoring run
            y0[-1] = 0.01  # initialize effector cell to 0.01 for scoring run
            t = (0, 20.)
            _, score = i.immune_network.run_graph(t, y0=y0, score=True)
            score_after.append(score[0])

        # During this run (single generation), the following evolution happens:
        # Graph 1: no change
        self.assertTrue(edges_before[0] == edges_after[0])
        self.assertTrue(score_before[0] == score_after[0])

        # Graph 2: t0 gets deleted and the (r0, c0) interaction gets deleted
        self.assertFalse(score_before[1] == score_after[1])

        # Graph 3: does not get selected for the next generation;
        #          instead, Graph 2 gets selected to replace it
        self.assertTrue(score_after[1] == score_after[2])

        # Graph 4: no change
        self.assertTrue(score_before[3] == score_after[3])

        # ------------------ #
        # Testing (2)
        # ------------------ #
        # Graph 2 has the greatest fitness after calling netPop.run()
        # Graph 3 has the lowest relative fitness after calling netPop.run()
        # This explains why Graph 2 replace Graph 3 during the selection step
        # This test is only a probabilistic test but it's better than nothing
        self.assertTrue(np.argmin(norm_rel_fit) == 2)
        self.assertTrue(np.argmax(norm_rel_fit) == 1)
        self.assertTrue(np.argmax(popfit[0]) == 1)

    def test_multiprocessing(self):
        """
        Tests that the number of individuals is the same before and after the
        run, which would not be the case if one of the threads didn't finish or
        if we fail to collect some of the results from the Queue.
        """
        numGen = 4
        netPop = graph_evolution.NetworkPopulation(num_generations=numGen, num_individuals=4)
        num_ind_before = copy.deepcopy(netPop.num_individuals)
        popfit = netPop.run(cpu=2)
        num_ind_after = netPop.num_individuals
        self.assertEqual(num_ind_before, num_ind_after)
        self.assertEqual(len(popfit), numGen)


if __name__ == '__main__':
    unittest.main()
