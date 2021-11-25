import immune_network
from immune_network import ImmuneGraph, Parasite
from scipy.integrate import solve_ivp
import numpy as np
import os
import multiprocessing as mp
from multiprocessing import Process, Queue
import copy
import dill as pickle
import sys
from tqdm import tqdm


def __x_in_range__(x, rng):
    if rng[0] <= x and x < rng[1]:
        return True
    else:
        return False


class NetworkEvolver():
    """
    Class represents a single individual in a genetic optimization algorithm.
    Contains an ImmuneGraph instance and metadata associated with mutating this
    graph during optimization. Class also contains methods for evolving the
    ImmuneGraph instance during optimization.
    """
    def __init__(self, **kwargs):
        self.immune_network = immune_network.ImmuneGraph()
        self.out_dir = kwargs.get('out_dir', os.getcwd())
        self.generation_id = 0  # counts generations
        self.fitness_history = []
        # mutation probabilities
        #  note that the sum of all but mut_prob must add to 1
        self.mut_prob = 0.05  # prob. of mutation per signal protein
        self.new_int_prob = 0.25  # probability of adding a new interaction
        self.del_int_prob = 0.25  # probability of deleting an interaction
        self.mut_int_prob = 0.3  # prob. of mutating an interaction strength
        self.dup_prob = 0.1  # probability of duplicating a protein
        self.del_prob = 0.1  # probability of deleting a protein

        probs = [self.new_int_prob, self.del_int_prob, self.mut_int_prob, self.dup_prob, self.del_prob]
        assert np.sum(probs) == 1.
        self.decision_boundaries = []
        for i in range(len(probs)):
            self.decision_boundaries.append(np.sum(probs[:i]))
        self.decision_boundaries.append(1.0)

    def __score__(self):
        y0 = self.immune_network.equilibrate()
        y0[0] = 0.5  # initialize parasite to 0.5 for scoring run
        y0[1] = 0.01  # initialize detecotr cell to 0.01 for scoring run
        y0[-1] = 0.01  # initialize effector cell to 0.01 for scoring run
        t = (0, 20.)
        _, score = self.immune_network.run_graph(t, y0=y0, score=True)
        return score

    def evolve_graph(self, DEBUG=False, DEBUG_decision=0., DEBUG_choice=''):
        mut_probability = self.mut_prob * self.immune_network.mutable_species
        if DEBUG:
            mut_probability = 1.
        if np.random.random() < mut_probability:
            decision = np.random.random()
            if DEBUG:
                decision = DEBUG_decision
            if __x_in_range__(decision, self.decision_boundaries[0:2]):
                self.immune_network.new_interaction(DEBUG_choice=DEBUG_choice)
            elif __x_in_range__(decision, self.decision_boundaries[1:3]):
                self.immune_network.delete_interaction(DEBUG_choice=DEBUG_choice)
            elif __x_in_range__(decision, self.decision_boundaries[2:4]):
                self.immune_network.mutate_interaction(DEBUG_choice=DEBUG_choice)
            elif __x_in_range__(decision, self.decision_boundaries[3:5]):
                self.immune_network.duplicate_protein(DEBUG_choice=DEBUG_choice)
            elif __x_in_range__(decision, self.decision_boundaries[4:6]):
                self.immune_network.delete_protein(DEBUG_choice=DEBUG_choice)
            else:
                print("Mutation probabilities may have changed")
                print(self.decision_boundaries)
                print(decision)
                raise ValueError

        self.generation_id += 1

    def get_next_gen(self, DEBUG=False):
        """
        Performs one evolution step, which includes probabilistically evolving
        the graph and then scoring the new graph. The DEBUG flag allows one to
        monitor the network structure.
        """
        if DEBUG:
            edges_before = copy.deepcopy(self.immune_network.graph.edges())

        self.evolve_graph(DEBUG=DEBUG)

        if DEBUG:
            edges_after = copy.deepcopy(self.immune_network.graph.edges())

        score = self.__score__()
        self.fitness_history.append(score[0])

        if DEBUG:
            return [edges_before, edges_after]

    def save(self, out_dir, id):
        file = os.path.join(out_dir, 'net'+str(id)+'.pkl')
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    def load(self, pik):
        with open(pik, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)


def __evo_func__(q, arr, idx, DEBUG=False):
    if not DEBUG:
        for i, net in enumerate(arr):
            net.get_next_gen()
            out = (idx+i, net.fitness_history[-1])
            q.put(out)
        return 0
    else:
        for i, net in enumerate(arr):
            net.get_next_gen()
            out = (idx+i, net.fitness_history[-1])
            q.append(out)
        return q


class NetworkPopulation():
    def __init__(self, **kwargs):
        self.num_generations = kwargs.get('num_generations', 600)
        self.num_individuals = kwargs.get('num_individuals', 1000)
        self.individuals = [NetworkEvolver() for _ in range(self.num_individuals)]

    def load(self, pik):
        with open(pik, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)

    def save(self, pik):
        with open(pik, 'wb') as f:
            pickle.dump(self, f)

    def __mkdir__(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def __save_gen__(self, gen_dir):
        for idx, i in enumerate(self.individuals):
            i.save(gen_dir, idx)
        self.save(os.path.join(gen_dir, 'Pop.pkl'))

    def __select__(self, individuals, fitnesses):
        num_individuals = len(individuals)

        # Compute relative fitness within this generation
        mean_fit = np.mean(fitnesses)
        rel_fit = np.divide(fitnesses, mean_fit)

        # sorts pop. by rel. fit. (biggest first)
        sort_idx = np.argsort(rel_fit)[::-1]
        norm_rel_fit = np.divide(rel_fit, max(rel_fit))

        new_pop = []
        new_pop_fitness = []
        # Select individuals until we have a population of the right size
        while len(new_pop) < num_individuals:
            fit_bound = np.random.random()
            # find all individuals with rel. fitness greater than fit_bound
            selection_pool = []
            for i in sort_idx:
                if norm_rel_fit[i] > fit_bound:
                    selection_pool.append(i)
            # choose new individual from selection_pool
            new_idx = np.random.choice(selection_pool)
            new_pop.append(copy.deepcopy(individuals[new_idx]))
            # Also record the fitness of the selected individual
            new_pop_fitness.append(fitnesses[new_idx])
        return new_pop, new_pop_fitness, norm_rel_fit

    def __score__(self, cpu, processors, interval, evo_func):
        if cpu > 1:
            q = Queue()
            processes = []
            for i in range(processors-1):
                arr = self.individuals[i*interval:(i+1)*interval]
                p = Process(target=evo_func, args=(q, arr, i*interval))
                processes.append(p)
                p.start()
            arr = self.individuals[(processors-1)*interval:]
            processes.append(Process(target=evo_func, args=(q, arr, (processors-1)*interval)))
            processes[-1].start()
            # Combine results from all processes
            fit_scores = []
            for r in range(self.num_individuals):
                fit_scores.append(q.get())
            for p in processes:
                p.join()  # wait for all processes to finish

        else:  # single thread version; useful for debugging
            q = []
            fit_scores = evo_func(q, self.individuals, 0, DEBUG=True)

        # Close queue and multiprocessing
        if cpu > 1:
            q.close()
            if sys.version_info[1] > 6:
                for p in processes:
                    p.close()

        return fit_scores

    def run(self, cpu=8, out_dir='', DEBUG=False):
        """
        Optionally multithreaded.
        """
        # Set up multiprocessing
        if DEBUG:
            cpu = 1
        processors = min(cpu, mp.cpu_count())
        interval = int(self.num_individuals / processors)
        # Set up output directory
        if out_dir == '':
            out_dir = os.path.join(os.getcwd(), 'output')
            self.__mkdir__(out_dir)
        # Iterate through generations
        population_fitness = []
        for g in tqdm(range(self.num_generations)):
            if not DEBUG:
                # Save population:
                gen_dir = os.path.join(out_dir, 'g'+str(g))
                self.__mkdir__(gen_dir)
                self.__save_gen__(gen_dir)
            # For this generation, evaluate individuals
            fit_scores = self.__score__(cpu, processors, interval, __evo_func__)

            fit_scores.sort(key=lambda tup: tup[0])  # sorts in place
            fit_scores = [el[1] for el in fit_scores]   # discards index

            # Perform evolutionary selection
            population_fitness.append([])
            new_pop, new_pop_fitness, norm_rel_fit =\
                self.__select__(self.individuals, fit_scores)
            self.individuals = new_pop
            population_fitness[-1].append(new_pop_fitness)

        if DEBUG:
            return population_fitness, norm_rel_fit
        else:
            return population_fitness


class CoEvolutionGraph(ImmuneGraph):
    """
    CoEvolutionGraph object stores a NetworkX graph representing a signaling
    network and a paired parasite.
    This object modifies several methods from the inhereited ImmuneGraph class.
    """
    def __init__(self, immnet, parasite):
        super().__init__()
        self.parasite = parasite
        # transfer all parameters from input immnet
        self.graph = immnet.graph
        self.cytokines = immnet.cytokines
        self.largest_cytokine_label = immnet.largest_cytokine_label
        self.receptors = immnet.receptors
        self.largest_receptor_label = immnet.largest_receptor_label
        self.transcription_factors = immnet.transcription_factors
        self.largest_tf_label = immnet.largest_tf_label
        self.mutable_species = immnet.mutable_species
        self.k_par_eff = immnet.k_par_eff
        self.k_det_par = immnet.k_det_par
        self.r_par = immnet.r_par
        self.k_on = immnet.k_on

    def run_graph(self, t_span, y0=np.array([]), score=False, DEBUG=False):
        """
        Overrides the run_graph() method from ImmuneGraph class.
        The main difference is that the effect of the parasite must be added to
        the derivative function definition.
        """
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

        def fun(t, y, debug=DEBUG):
            # t is irrelevant unless there is an explicit dependence on time
            dydt = np.zeros(len(y))
            # infectious agent
            if y[0] < 1E-4:  # infection is considered cleared
                dydt[0] = -y[0]
            else:
                #              effector kills agent   + logistic growth
                dydt[0] += -self.k_par_eff*y[-1]*y[0] + self.r_par*y[0]*(1.-y[0])

            # ----------------------------------------
            # Effect of parasite on host protein
            # ----------------------------------------
            if y[0] > 1E-4:  # parasite must be present to have effect
                target = self.parasite.interaction_node.startswith('r')
                # target must also be in the host network
                if target in self.receptors or target in self.transcription_factors:
                    if target.startswith('r'):  # receptor
                        idx_i = self.__find_index__(self.receptors, target) + 2 + num_cytokines
                        idx_a = self.__find_index__(self.receptors, target) + 2 + num_cytokines + num_receptors
                    else:  # transcription factor
                        idx_i = self.__find_index__(self.transcription_factors, target) + 2 + num_cytokines + 2*num_receptors
                        idx_a = self.__find_index__(self.transcription_factors, target) + 2 + num_cytokines + 2*num_receptors + num_tfs
                    if self.parasite.interaction_strength >= 0:
                        # if positive interaction then host protein set to fully active
                        y[idx_i] = 0.
                        y[idx_a] = 1.
                    else:
                        # if negative interaction then host protein fully inactive
                        y[idx_i] = 1.
                        y[idx_a] = 0.
            # ----------------------------------------

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
