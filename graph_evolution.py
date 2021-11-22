import immune_network
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

    def evolve_graph(self, DEBUG=False, DEBUG_decision=0.):
        mut_probability = self.mut_prob * self.immune_network.mutable_species
        if DEBUG:
            mut_probability = 1.
        if np.random.random() < mut_probability:
            decision = np.random.random()
            if DEBUG:
                decision = DEBUG_decision
            if __x_in_range__(decision, self.decision_boundaries[0:2]):
                self.immune_network.new_interaction()
            elif __x_in_range__(decision, self.decision_boundaries[1:3]):
                self.immune_network.delete_interaction()
            elif __x_in_range__(decision, self.decision_boundaries[2:4]):
                self.immune_network.mutate_interaction()
            elif __x_in_range__(decision, self.decision_boundaries[3:5]):
                self.immune_network.duplicate_protein()
            elif __x_in_range__(decision, self.decision_boundaries[4:6]):
                self.immune_network.delete_protein()
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

    def run(self, cpu=8, out_dir='', DEBUG=False):
        if DEBUG:
            cpu = 1
        processors = min(cpu, mp.cpu_count())
        interval = int(self.num_individuals / processors)
        population_fitness = []
        if out_dir == '':
            out_dir = os.path.join(os.getcwd(), 'output')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        for g in tqdm(range(self.num_generations)):
            if not DEBUG:
                # Save population:
                gen_dir = os.path.join(out_dir, 'g'+str(g))
                if not os.path.exists(gen_dir):
                    os.makedirs(gen_dir)
                for idx, i in enumerate(self.individuals):
                    i.save(gen_dir, idx)
                self.save(os.path.join(gen_dir, 'Pop.pkl'))
            # Evaluate individuals
            if cpu > 1:
                q = Queue()
                processes = []
                for i in range(processors-1):
                    arr = self.individuals[i*interval:(i+1)*interval]
                    p = Process(target=__evo_func__, args=(q, arr, i*interval))
                    processes.append(p)
                    p.start()
                arr = self.individuals[(processors-1)*interval:]
                processes.append(Process(target=__evo_func__, args=(q, arr, (processors-1)*interval)))
                processes[-1].start()
                # Combine results from all processes
                fit_scores = []
                for r in range(self.num_individuals):
                    fit_scores.append(q.get())
                for p in processes:
                    p.join()  # wait for all processes to finish

            else:  # single thread version; useful for debugging
                q = []
                fit_scores = __evo_func__(q, self.individuals, 0, DEBUG=True)

            fit_scores.sort(key=lambda tup: tup[0])  # sorts in place
            fit_scores = [el[1] for el in fit_scores]   # discards index

            # Close queue and multiprocessing
            if cpu > 1:
                q.close()
                if sys.version_info[1] > 6:
                    for p in processes:
                        p.close()

            # Compute relative fitness within this generation
            mean_fit = np.mean(fit_scores)
            rel_fit = np.divide(fit_scores, mean_fit)
            sort_idx = np.argsort(rel_fit)[::-1]  # sorts pop. by rel. fit. (biggest first)
            norm_rel_fit = np.divide(rel_fit, max(rel_fit))

            # Select next generation
            new_pop = []
            # Add new row for fitness of selected individuals
            population_fitness.append([])
            # Select individuals until we have a population of the right size
            while len(new_pop) < self.num_individuals:
                fit_bound = np.random.random()
                # find all individuals with rel. fitness greater than fit_bound
                selection_pool = []
                for i in sort_idx:
                    if norm_rel_fit[i] > fit_bound:
                        selection_pool.append(i)
                # choose new individual from selection_pool
                new_idx = np.random.choice(selection_pool)
                new_pop.append(copy.deepcopy(self.individuals[new_idx]))
                # Also record the fitness of the selected individual
                population_fitness[-1].append(fit_scores[new_idx])
            self.individuals = new_pop

        if DEBUG:
            return population_fitness, norm_rel_fit
        else:
            return population_fitness
