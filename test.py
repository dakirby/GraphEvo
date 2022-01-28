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
        final_vec = sim.y[:, -1]
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
        G = ne.immune_network.graph
        # -------------------- #
        # duplicate_protein()
        # -------------------- #
        ne.evolve_graph(DEBUG=True, DEBUG_decision=0.85, DEBUG_choice='t0')
        wE0 = G.get_edge_data('t0', 'E')
        wr0 = G.get_edge_data('t0', 'r0')
        wE1 = G.get_edge_data('t1', 'E')
        wr1 = G.get_edge_data('t1', 'r0')
        self.assertEqual(wE0, wE1)
        self.assertEqual(wr0, wr1)

        # -------------------- #
        # new_interaction()
        # -------------------- #
        test_ed = ('t1', 't0')
        ne.evolve_graph(DEBUG=True, DEBUG_decision=0.1, DEBUG_choice=test_ed)
        new_int = copy.deepcopy(G.get_edge_data(*test_ed)['weight'])
        self.assertTrue(isinstance(new_int, float))
        # there is an interaction assigned

        # -------------------- #
        # mutate_interaction()
        # -------------------- #
        test_ed = ('t1', 't0')
        ne.evolve_graph(DEBUG=True, DEBUG_decision=0.7, DEBUG_choice=test_ed)
        wNew = G.get_edge_data(*test_ed)['weight']
        self.assertNotEqual(wNew, new_int)  # interaction has changed

        # -------------------- #
        # delete_interaction()
        # -------------------- #
        test_ed = ('E', 't1')
        self.assertTrue(test_ed in G.edges())
        ne.evolve_graph(DEBUG=True, DEBUG_decision=0.3, DEBUG_choice=test_ed)
        self.assertFalse(test_ed in G.edges())

        # -------------------- #
        # delete_protein()
        # -------------------- #
        ne.evolve_graph(DEBUG=True, DEBUG_decision=0.95, DEBUG_choice='t1')
        self.assertFalse(('r0', 't1') in G.edges())
        self.assertFalse(('t0', 't1') in G.edges())


class TestNetworkPopulationMethods(unittest.TestCase):
    """
    Tests the run() method to:
    1) ensure that each graph evolves independently
    2) multiprocessing correctly repeats evolution behaviour
    """

    def test_independent_and_selection(self):
        netPop = graph_evolution.NetworkPopulation(num_generations=1, num_individuals=4)

        for i in range(4):
            netPop.individuals[i].immune_network.duplicate_protein()
        netPop.individuals[1].immune_network.delete_interaction()
        edges_before = []
        score_before = []
        weights_before = [[] for _ in range(4)]
        for q, i in enumerate(netPop.individuals):
            edges_before.append(i.immune_network.graph.edges())
            for ed in i.immune_network.graph.edges():
                weights_before[q].append(i.immune_network.graph.get_edge_data(*ed))

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
        weights_after = [[] for _ in range(4)]
        for q, i in enumerate(netPop.individuals):
            edges_after.append(i.immune_network.graph.edges())
            for ed in i.immune_network.graph.edges():
                weights_after[q].append(i.immune_network.graph.get_edge_data(*ed))

            y0 = i.immune_network.equilibrate()
            y0[0] = 0.5  # initialize parasite to 0.5 for scoring run
            y0[1] = 0.01  # initialize detecotr cell to 0.01 for scoring run
            y0[-1] = 0.01  # initialize effector cell to 0.01 for scoring run
            t = (0, 20.)
            _, score = i.immune_network.run_graph(t, y0=y0, score=True)
            score_after.append(score[0])

        # Tests that if the score remains the same then the edges were
        # unchanged, otherwise the edges changed
        u = [score_before[q]==score_after[q] for q in range(len(score_before))]
        unchanged_test = []
        for idx, val in enumerate(u):
            test = (edges_before[idx] == edges_after[idx]) and (weights_before[idx] == weights_after[idx])
            unchanged_test.append(test == val)
        self.assertTrue(np.all(unchanged_test))

        return 1

    def test_multiprocessing(self):
        """
        Tests that the number of individuals is the same before and after the
        run, which would not be the case if one of the threads didn't finish or
        if we fail to collect some of the results from the Queue.
        """
        numGen = 4
        numInd = 50
        netPop = graph_evolution.NetworkPopulation(num_generations=numGen, num_individuals=numInd)
        num_ind_before = len(netPop.individuals)
        popfit = netPop.run(cpu=2)
        num_ind_after = len(netPop.individuals)
        self.assertEqual(num_ind_before, num_ind_after)
        self.assertEqual(len(popfit), numGen)


class TestCoEvolutionGraphMethods(unittest.TestCase):

    def test_run_graph(self):
        """
        Tests that:
        1) equilibration does not include the effect of the parasite
        2) ODE integration with the effect of a parasite agrees with results
           computed in Mathematica.
        """
        parasite = immune_network.Parasite()
        parasite.new_interaction_protein(global_ImmuneGraph, DEBUG_decision='t0')
        coev = immune_network.CoEvolutionGraph(global_ImmuneGraph, parasite)
        # ---------------
        # Test 1
        # ---------------
        t = (0., 20.)
        y0 = coev.equilibrate()
        cyt_eq = y0[2] - 24.6838  # cytokine equilibrium
        rec_eq = y0[3] - 0.03412  # receptor equilibrium
        tfr_eq = y0[5] - 6.7E-24  # transcription factor equilibrium
        eff_eq = y0[-1] - 1.0000  # effector cell equilibrium
        y0Test = np.array([cyt_eq, rec_eq, tfr_eq, eff_eq])
        self.assertTrue(np.all(y0Test < 0.0001))
        # ---------------
        # Test 2
        # ---------------
        y0[0] = 0.5  # set parasite to 0.5
        y0[1] = 0.01  # set detector cell to 0.01
        y0[-1] = 0.01  # set effector cell to 0.01
        sim, score = coev.run_graph(t, y0=y0, score=True)
        final_vec = sim.y[:, -1]

        # Check that transcription factor is always active due to parasite
        self.assertTrue(np.all(sim.y[-2, :]-1. < 0.0001))

        # Check that species trajectories agree with solution from Mathematica
        Math_final_vec = [0.00421984, 0.08538, 26.8394, 0.0314334, 0.968567, 0., 1., 1.]
        rel_err = np.subtract(Math_final_vec, final_vec)
        self.assertTrue(np.all(rel_err < 0.01))

    def test_run_CoEvolutionPopulation(self):
        KWARGS = {'num_generations': 3, 'num_individuals': 10, 'num_parasites': 2}
        netPop = graph_evolution.CoEvolutionPopulation(**KWARGS)
        fit_hist = netPop.run(cpu=2)


if __name__ == '__main__':
    unittest.main()
