import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import copy


class ImmuneGraph():
    """
    ImmuneGraph object stores a NetworkX graph representing a signaling network
    as well as several methods for modifying and analysing this graph.
    """
    def __init__(self):
        # Build minimal singaling network graph
        self.graph = nx.Graph()
        # Detector cell releases at least one cytokine
        self.graph.add_edge('D', 'c0', weight=self.new_production_rate())
        # Cytokine binds to receptor
        self.graph.add_edge('c0', 'r0', weight=self.new_affinity())
        # Receptor has at least one output
        self.graph.add_edge('r0', 't0', weight=self.new_production_rate())
        # Effector cell has at least one connection to TF
        self.graph.add_edge('t0', 'E', weight=self.new_production_rate())
        # Record several useful attributes
        self.cytokines = ['c0']
        self.largest_cytokine_label = 0
        self.receptors = ['r0']
        self.largest_receptor_label = 0
        self.transcription_factors = ['t0']
        self.largest_tf_label = 0
        self.mutable_species = len(self.cytokines) + len(self.receptors) + len(self.transcription_factors)
        # Fixed ODE parameters
        self.k_par_eff = 1.0
        self.k_det_par = 1.0
        self.r_par = 0.8  # 0.8 hr^-1 is typical infectious agent growth rate
        self.k_on = 1.0  # CHECK UNITS

    # Define functions necessary to expand graph
    def new_affinity(self):
        return np.exp(np.random.normal())

    def new_production_rate(self):
        return np.exp(np.random.normal())

    def new_interaction_strength(self):
        return np.random.normal()

    def choose_cytokine(self):
        return np.random.choice(self.cytokines)

    def choose_receptor(self):
        return np.random.choice(self.receptors)

    def choose_protein(self):
        proteins = self.cytokines + self.receptors + self.transcription_factors
        return np.random.choice(proteins)

    # Plotting functions
    def draw_graph(self, layout=graphviz_layout):
        # build colormap reflecting type of chemical species
        cmap = []
        for n in self.graph.nodes():
            if n == 'D':
                cmap.append('r')
            elif n == 'E':
                cmap.append('g')
            elif n.startswith('c'):
                cmap.append('b')
            elif n.startswith('r'):
                cmap.append('c')
            else:
                cmap.append('k')
        # plot network
        nx.draw(self.graph, pos=layout(self.graph), with_labels=False, node_color=cmap)
        plt.show()

    # Evolution Functions
    def new_interaction(self):
        prot = self.choose_protein()
        if prot.startswith('c'):  # cytokine
            r_cnx = [ed[1] for ed in self.graph.edges(prot) if ed[1].startswith('r')]
            unconnected_r = [r for r in self.receptors if r not in r_cnx]
            if unconnected_r != []:  # cytokine is not connected to all receptors
                rec = np.random.choice(unconnected_r)
                self.graph.add_edge(prot, rec, weight=self.new_affinity())

        if prot.startswith('r'):  # receptor
            c_cnx = [ed[1] for ed in self.graph.edges(prot) if ed[1].startswith('c')]
            unconnected_c = [c for c in self.cytokines if c not in c_cnx]
            tf_cnx = [ed[1] for ed in self.graph.edges(prot) if ed[1].startswith('t')]
            unconnected_tf = [t for t in self.transcription_factors if t not in tf_cnx]
            if unconnected_c != [] or unconnected_tf != []:
                # receptor is not fully connected
                node = np.random.choice(unconnected_c + unconnected_tf)
                if node.startswith('c'):
                    self.graph.add_edge(prot, node, weight=self.new_affinity())
                else:
                    w = self.new_production_rate()
                    self.graph.add_edge(prot, node, weight=w)

        if prot.startswith('t'):  # transcription factor
            r_cnx = [ed[1] for ed in self.graph.edges(prot) if ed[1].startswith('r')]
            unconnected_r = [r for r in self.receptors if r not in r_cnx]
            tf_cnx = [ed[1] for ed in self.graph.edges(prot) if ed[1].startswith('t')]
            unconnected_tf = [t for t in self.transcription_factors if t not in tf_cnx and t != prot]
            E_cnx = [ed[1] for ed in self.graph.edges(prot) if ed[1] == 'E']
            if E_cnx == []:
                unconnected_E = ['E']
            else:
                unconnected_E = []
            if unconnected_r != [] or unconnected_tf != [] or unconnected_E != []:
                # tf is not fully connected
                node = np.random.choice(unconnected_r + unconnected_tf + unconnected_E)
                if node.startswith('r'):
                    self.graph.add_edge(prot, node, weight=self.new_production_rate())
                else:
                    w = self.new_interaction_strength()
                    self.graph.add_edge(prot, node, weight=w)

    def delete_interaction(self):
        prot = self.choose_protein()
        # Decetor cell to cytokine does not count as an interaction
        allowed_edges = [ed for ed in self.graph.edges(prot) if 'D' not in ed]
        if allowed_edges != []:
            ed = allowed_edges[np.random.choice(list(range(len(allowed_edges))))]
            self.graph.remove_edge(*ed)

    def mutate_interaction(self):
        prot = self.choose_protein()
        edges = self.graph.edges(prot)
        if len(edges) != 0:
            ed = edges[np.random.choice(list(range(len(edges))))]
            w = self.graph.get_edge_data(*ed)['weight']
            w = w * (0.5 + np.random.random())
            self.graph[ed[0]][ed[1]]['weight'] = w
        # otherwise just bad luck, skip to avoid infinite searching for edges

    def duplicate_protein(self):
        prot = self.choose_protein()
        edges = self.graph.edges(prot)
        prot_type = prot[0]
        self.mutable_species += 1
        if prot_type == 'c':
            label = 'c' + str(self.largest_cytokine_label + 1)
            self.cytokines.append(label)
            self.largest_cytokine_label += 1
        elif prot_type == 'r':
            label = 'r' + str(self.largest_receptor_label + 1)
            self.receptors.append(label)
            self.largest_receptor_label += 1
        else:  # prot_type == 't'
            label = 't' + str(self.largest_tf_label + 1)
            self.transcription_factors.append(label)
            self.largest_tf_label += 1

        for ed in edges:
            w = self.graph.get_edge_data(*ed)['weight']
            self.graph.add_edge(label, ed[1], weight=w)

    def delete_protein(self):
        prot = self.choose_protein()
        prot_type = prot[0]
        self.mutable_species -= 1
        self.graph.remove_node(prot)
        if prot_type == 'c':
            idx = self.__find_index__(self.cytokines, prot)
            self.cytokines.pop(idx)
        elif prot_type == 'r':
            idx = self.__find_index__(self.receptors, prot)
            self.receptors.pop(idx)
        else:
            idx = self.__find_index__(self.transcription_factors, prot)
            self.transcription_factors.pop(idx)

    def __find_index__(self, arr, el):
        for idx, a in enumerate(arr):
            if a == el:
                return idx
        print(arr, el)
        raise KeyError

    # Build ODE system based on graph
    def run_graph(self, t_span, y0=np.array([]), score=False, DEBUG=False):
        # Build the ODE integrator
        num_cytokines = len(self.cytokines)
        num_receptors = len(self.receptors)
        num_tfs = len(self.transcription_factors)
        if y0.size == 0:
            # initial species concentrations
            init_agent = [0.5]
            init_detector = [0.01]
            init_effector = [0.01]
            init_cytokines = [0. for _ in range(num_cytokines)]
            # inactive species plus active species
            init_receptors = [1. for _ in range(num_receptors)] + [0. for _ in range(num_receptors)]
            init_tfs = [1. for _ in range(num_tfs)] + [0. for _ in range(num_tfs)]
            y0 = init_agent + init_detector + init_cytokines + init_receptors + init_tfs + init_effector
            y0 = np.array(y0)
        else:
            assert y0.shape[0] == 3 + num_cytokines + 2*(num_receptors + num_tfs)

        num_species = len(y0)

        def fun(t, y, debug=DEBUG):
            # t is irrelevant unless there is an explicit dependence on time
            dydt = np.zeros(num_species)
            # infectious agent
            #              effector kills agent   + logistic growth
            dydt[0] += -self.k_par_eff*y[-1]*y[0] + self.r_par*y[0]*(1.-y[0])

            # detector cell activation
            dydt[1] += self.k_det_par*y[0]*y[1]*(1.-y[1])

            # cytokine production
            for i, ed in enumerate(self.graph.edges('D'), start=2):
                k_prod = self.graph.get_edge_data('D', ed[1])['weight']
                dydt[i] += k_prod*y[1]

            # cytokine binding dynamics
            for i, cyt in enumerate(self.cytokines, start=2):
                for ed in self.graph.edges(cyt):
                    if ed[1] != 'D':
                        r_offset = self.__find_index__(self.receptors, ed[1]) + 2 + num_cytokines
                        k_unbind = self.graph.get_edge_data(cyt, ed[1])['weight']
                        dydt[i] += -self.k_on*y[i]*y[r_offset]  # binding to inactive receptors
                        dydt[i] += k_unbind*y[r_offset + num_receptors]  # unbinding from active receptors

            # receptor binding dynamics
            for rec in self.receptors:
                r_offset = self.__find_index__(self.receptors, rec) + 2 + num_cytokines
                for ed in self.graph.edges(rec):
                    if ed[1].startswith('c'): # cytokine edge, not TF edge
                        cyt = ed[1]
                        c_idx = 2 + self.__find_index__(self.cytokines, cyt)
                        k_unbind = self.graph.get_edge_data(rec, cyt)['weight']

                        # binding to inactive receptors
                        dydt[r_offset] += -self.k_on*y[c_idx]*y[r_offset]
                        dydt[r_offset+num_receptors] += self.k_on*y[c_idx]*y[r_offset]
                         # unbinding from active receptors
                        dydt[r_offset] += k_unbind*y[r_offset+num_receptors]
                        dydt[r_offset+num_receptors] += -k_unbind*y[r_offset+num_receptors]

            # transcription factor dynamics and effector activity
            for tf in self.transcription_factors:
                # inactive tf index
                tf_idx = self.__find_index__(self.transcription_factors, tf) + 2 + num_cytokines + 2*num_receptors
                for ed in self.graph.edges(tf):
                    if ed[1].startswith('r'):  # receptor
                        k_prod = self.graph.get_edge_data(tf, ed[1])['weight']
                        # get active receptor species index
                        r_idx = self.__find_index__(self.receptors, ed[1]) + 2 + num_cytokines + num_receptors
                        # production dynamics
                        dydt[tf_idx] += -k_prod*y[r_idx]*y[tf_idx]
                        dydt[tf_idx+num_tfs] += k_prod*y[r_idx]*y[tf_idx]
                    elif ed[1].startswith('t'):  # TF interaction
                        k_ij = self.graph.get_edge_data(tf, ed[1])['weight']
                        # get inactive second TF index
                        tf2_idx = self.__find_index__(self.transcription_factors, ed[1])
                        tf2_idx += 2 + num_cytokines + 2*num_receptors
                        # interaction dynamics
                        if k_ij > 0:  # promotion
                            dydt[tf_idx] += -k_ij*y[tf2_idx+num_tfs]*y[tf_idx]
                            dydt[tf_idx+num_tfs] += k_ij*y[tf2_idx+num_tfs]*y[tf_idx]
                        else:  # inhibition
                            dydt[tf_idx] += -k_ij*y[tf2_idx+num_tfs]*y[tf_idx+num_tfs]
                            dydt[tf_idx+num_tfs] += k_ij*y[tf2_idx+num_tfs]*y[tf_idx+num_tfs]
                    elif ed[1] == 'E':
                        k_act = self.graph.get_edge_data(tf, ed[1])['weight']
                        dydt[-1] += k_act*y[tf_idx+num_tfs]*y[-1]*(1.-y[-1])
                        # this should normalize effector activity?
                    else:
                        print('Something has mapped this TF to an invalid species')
                        print(ed)
                        raise KeyError
                        return 0

            return dydt

        # Solve the ODE numerically
        sol = solve_ivp(fun, t_span, y0)

        # compute fitness of this solution
        if score:
            Peff_pre = sol.y[-1][0]
            Peff_post = sol.y[-1][-1]
            auc = np.trapz(sol.y[0], x=sol.t)
            volume = 1. / (t_span[1]-t_span[0])
            normalized_auc = auc * volume
            fitness_exp = Peff_pre + Peff_post + normalized_auc
            Whost = np.exp(-fitness_exp)
            Wagent = np.exp(-(2+volume*(1-auc))) # typo in Graham paper? I think this is correct...
            return sol, (Whost, Wagent)
        else:
            return sol

    def equilibrate(self):
        """
        Initialize all species to 0.5 except parasite set to 0, let run for a
        while and then return the state of the system.
        """
        num_cytokines = len(self.cytokines)
        num_receptors = len(self.receptors)
        num_tfs = len(self.transcription_factors)

        t_span = (0., 30.)
        y0_len = 3 + num_cytokines + 2*(num_receptors + num_tfs)
        y0 = [0.5 for _ in range(y0_len)]
        y0[0] = 0.
        y0 = np.array(y0)
        sol = self.run_graph(t_span, y0=y0)
        return sol.y[:, -1]

    def plot_simulation(self, tspan):
        sol = self.run_graph(tspan)
        agent_density = sol.y[0]
        effector_activity = sol.y[-1]
        detector_activity = sol.y[1]
        fig, ax = plt.subplots()
        plt.plot(sol.t, agent_density, 'r', label='Infectious Agent')
        plt.plot(sol.t, effector_activity, 'k', label='Effector Cell')
        plt.plot(sol.t, detector_activity, 'b', label='Detector Cell')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Activity (%)')
        plt.xticks([])
        plt.yticks([])

        plt.show()
