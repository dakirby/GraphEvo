import immune_network
from immune_network import ImmuneGraph, Parasite, CoEvolutionGraph
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
        self.immune_network = kwargs.get('immune_network', immune_network.ImmuneGraph())
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

    def get_next_gen(self, DEBUG=False, RETURN_SCORE=False, COEV=False):
        """
        Performs one evolution step, which includes probabilistically evolving
        the graph and then scoring the new graph. The DEBUG flag allows one to
        monitor the network structure.
        """
        if DEBUG:
            edges_before = copy.deepcopy(self.immune_network.graph.edges())

        self.evolve_graph(DEBUG=DEBUG)

        if COEV:
            # Give the parasites a chance to evolve
            if np.random.random() < self.immune_network.parasite_mut_rate:
                self.immune_network.parasite.mutate()

        if DEBUG:
            edges_after = copy.deepcopy(self.immune_network.graph.edges())

        score = self.__score__()
        self.fitness_history.append(score[0])

        if RETURN_SCORE:
            if COEV:
                return score, self.immune_network.parasite
            else:
                return score
        elif DEBUG:
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


def __coevo_func__(q, arr, idx, DEBUG=False):
    individuals, parasites = arr  # Unpack arr first
    if not DEBUG:
        for i, net in enumerate(individuals):
            net.parasite = parasites[i]
            score, parasite = net.get_next_gen(RETURN_SCORE=True, COEV=True)
            out = (idx+i, score, parasite)
            q.put(out)
        return 0
    else:
        for i, net in enumerate(arr):
            net.parasite = parasites[i]
            score, parasite = net.get_next_gen(RETURN_SCORE=True, COEV=True)
            out = (idx+i, score, parasite)
            q.append(out)
        return q


class NetworkPopulation():
    """
    Contains a population of NetworkEvolver instances which are iteratively
    evolved according to an evolutionary algorithm using fitness scores.
    The only methods available to the user are load(), save(), and run().
    """
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

    def __select__(self, individuals, fitnesses, **kwargs):
        num_individuals = kwargs.get('num_individuals', len(individuals))

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


class CoEvolutionPopulation(NetworkPopulation):
    """
    Contains a population of NetworkEvolver instances and a population of
    Parasite instances, both of which are iteratively evolved according to an
    evolutionary algorithm using fitness scores.
    The only methods available to the user are load(), save(), and run().
    """
    def __init__(self, **kwargs):
        self.num_generations = kwargs.get('num_generations', 600)
        self.num_individuals = kwargs.get('num_individuals', 1000)
        self.num_parasites = kwargs.get('num_parasites', 10)

        self.parasites = [Parasite() for _ in range(self.num_parasites)]
        self.pairings = np.random.choice(self.num_parasites, size=self.num_individuals)
        self.individuals = []
        for i in range(self.num_individuals):
            p = self.parasites[self.pairings[i]]
            imm = immune_network.ImmuneGraph()
            new_immune_network = immune_network.CoEvolutionGraph(imm, p)
            KWARG = {'immune_network': copy.deepcopy(new_immune_network)}
            self.individuals.append(NetworkEvolver(**KWARG))

    def __score__(self, cpu, processors, interval, evo_func):
        """
        Overwrites __score__() from NetworkPopulation class to enable the
        Parasites to be passed to each Process along with the individuals.
        """
        if cpu > 1:
            q = Queue()
            processes = []
            for i in range(processors-1):
                ind_arr = self.individuals[i*interval:(i+1)*interval]
                idxs = self.pairings[i*interval:(i+1)*interval]
                par_arr = [copy.deepcopy(self.parasites[i]) for i in idxs]
                arr = [ind_arr, par_arr]
                p = Process(target=evo_func, args=(q, arr, i*interval))
                processes.append(p)
                p.start()
            last_idx = (processors-1)*interval
            ind_arr = self.individuals[last_idx:]
            idxs = self.pairings[(processors-1)*interval:]
            par_arr = [copy.deepcopy(self.parasites[i]) for i in idxs]
            arr = [ind_arr, par_arr]

            processes.append(Process(target=evo_func, args=(q, arr, last_idx)))
            processes[-1].start()
            # Combine results from all processes
            fit_scores = []
            mutated_parasites = []
            for r in range(self.num_individuals):
                idx, score, parasite = q.get()
                fit_scores.append((idx, score))
                mutated_parasites.append(parasite)
            for p in processes:
                p.join()  # wait for all processes to finish

        else:  # single thread version; useful for debugging
            q = evo_func(q, self.individuals, 0, DEBUG=True)
            idx = q[:, 0]
            score = q[:, 1]
            mutated_parasites = q[:, 2]
            fit_scores = list(zip(idx, score))

        # Close queue and multiprocessing
        if cpu > 1:
            q.close()
            if sys.version_info[1] > 6:
                for p in processes:
                    p.close()

        return fit_scores, mutated_parasites

    def run(self, cpu=8, out_dir='', DEBUG=False):
        """
        Overwrites the run() function from the NetworkPopulation class so that
        the Parasites are also evolved during iteration of the evo algorithm.
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
        parasite_fitness = []
        for g in tqdm(range(self.num_generations)):
            if not DEBUG:
                # Save population:
                gen_dir = os.path.join(out_dir, 'g'+str(g))
                self.__mkdir__(gen_dir)
                self.__save_gen__(gen_dir)
            # For this generation, evaluate individuals
            fit_scores, mutated_parasites = self.__score__(cpu, processors, interval, __coevo_func__)

            # split and sort scores
            network_scores = [(el[0], el[1][0]) for el in fit_scores]
            network_scores.sort(key=lambda tup: tup[0])  # aligns scores with self.individuals
            network_scores = [el[1] for el in network_scores]

            parasite_scores = [(el[0], el[1][1]) for el in fit_scores]
            parasite_scores.sort(key=lambda tup: tup[0])  # aligns scores with self.individuals
            parasite_scores = [el[1] for el in parasite_scores]

            # mean_parasite_scores = [[] for _ in range(self.num_parasites)]
            # for idx, s in enumerate(parasite_scores):
            #     mean_parasite_scores[self.pairings[idx]].append(s)
            # for i in range(self.num_parasites):
            #     mean_parasite_scores[i] = np.mean(mean_parasite_scores[i])

            # Perform evolutionary selection on networks
            population_fitness.append([])
            new_pop, new_pop_fitness, norm_rel_fit =\
                self.__select__(self.individuals, network_scores)
            self.individuals = new_pop
            population_fitness[-1].append(new_pop_fitness)

            # Perform evolutionary selection on parasites
            parasite_fitness.append([])
            slct_kw = {'num_individuals': self.num_parasites}
            new_para_pop, new_para_pop_fitness, _ =\
                self.__select__(mutated_parasites, parasite_scores, **slct_kw)
            self.parasites = new_para_pop
            parasite_fitness[-1].append(new_para_pop_fitness)

            # Assign new parasites to each individual
            self.pairings = np.random.choice(self.num_parasites, size=self.num_individuals)
            for idx, pair_idx in enumerate(self.pairings):
                target = self.parasites[pair_idx]
                self.individuals[idx].immune_network.parasite = target

        if DEBUG:
            return population_fitness, parasite_fitness, norm_rel_fit
        else:
            return population_fitness, parasite_fitness
