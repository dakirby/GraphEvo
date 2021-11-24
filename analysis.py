import os
import immune_network
import graph_evolution
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx


def unnest(d, keys=[]):
    """
    Handy function to turn a dict into a tuple of tuples so that it is hashable
    and can therefore be used as a dictionary key.
    Source: https://stackoverflow.com/questions/57627575/convert-a-nested-dictionary-into-list-of-tuples
    """
    result = []
    for k, v in d.items():
        if isinstance(v, dict):
            result.extend(unnest(v, keys + [k]))
        else:
            result.append(tuple(keys + [k, v]))
    return tuple(result)


def plot_perplexity(outdir):
    """
    Counts the number of unique graphs in each generation and plots the
    perplexity over all generations. This plot is saved to outdir, as well as
    the dictionary of unique graphs and their abundances per generation.
    """
    generations = [int(dir[1:]) for dir in os.listdir(outdir) if dir.startswith('g')]
    num_generations = max(generations)
    individuals = [int(dir[3:-4]) for dir in os.listdir(outdir + os.sep + 'g0') if dir.startswith('net')]
    num_individuals = max(individuals)

    diversity = [{} for _ in range(num_generations)]

    # Build a dictionary of unique graphs
    # and count the number of occurences of each
    for g in tqdm(range(num_generations)):
        for i in tqdm(range(num_individuals), leave=False):
            ind = graph_evolution.NetworkEvolver()
            fname = os.path.join(outdir, 'g' + str(g), 'net' + str(i) + '.pkl')
            ind.load(fname)
            G = ind.immune_network.graph
            d = nx.to_dict_of_dicts(G)
            key = unnest(d)  # makes dict a hashable object
            if key not in list(diversity[g].keys()):
                diversity[g].update({key: 1})
            else:
                diversity[g][key] += 1
    np.save(outdir + os.sep + 'diversity.npy', diversity)

    # Compute the discrete probability of each unique graph
    discrete_prob = [[] for _ in range(num_generations)]
    for g in range(num_generations):
        for key in diversity[g].keys():
            discrete_prob[g].append(diversity[g][key] / num_individuals)
    # Compute the entropy and perplexity for each generation
    entropy = []
    for g in range(num_generations):
        ent = [-p*np.log2(p) for p in discrete_prob[g]]
        entropy.append(np.sum(ent))
    perplexity = np.exp(entropy)

    # Plot perplexity
    fig, ax = plt.subplots()
    ax.plot(perplexity, linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Perplexity')
    plt.savefig(outdir + os.sep + 'perplexity.pdf')


def plot_mean_connectivity(outdir):
    """
    Computes the average node degree for each class of nodes in the graph:
    cytokines, receptors, transcription factors, and the effector cell.
    Saves a graph of these quantities over all generations into outdir.
    """
    generations = [int(dir[1:]) for dir in os.listdir(outdir) if dir.startswith('g')]
    num_generations = max(generations)
    individuals = [int(dir[3:-4]) for dir in os.listdir(outdir + os.sep + 'g0') if dir.startswith('net')]
    num_individuals = max(individuals)

    cyt_conn = np.zeros((num_individuals, num_generations))
    rec_conn = np.zeros((num_individuals, num_generations))
    tf_conn = np.zeros((num_individuals, num_generations))
    E_conn = np.zeros((num_individuals, num_generations))

    for g in tqdm(range(num_generations)):
        for i in tqdm(range(num_individuals), leave=False):
            ind = graph_evolution.NetworkEvolver()
            fname = os.path.join(outdir, 'g' + str(g), 'net' + str(i) + '.pkl')
            ind.load(fname)
            G = ind.immune_network.graph
            c_sum, c_tot, r_sum, r_tot, tf_sum, tf_tot, E_sum, E_tot = (0, 0, 0, 0, 0, 0, 0, 0)
            for n in G.nodes():
                if n.startswith('c'):  # cytokine
                    c_sum += G.degree(n)
                    c_tot += 1
                if n.startswith('r'):  # receptor
                    r_sum += G.degree(n)
                    r_tot += 1
                if n.startswith('t'):  # cytokine
                    tf_sum += G.degree(n)
                    tf_tot += 1
                if n.startswith('E'):  # cytokine
                    E_sum += G.degree(n)
                    E_tot += 1

            cyt_conn[i][g] = c_sum / c_tot
            rec_conn[i][g] = r_sum / r_tot
            tf_conn[i][g] = tf_sum / tf_tot
            E_conn[i][g] = E_sum / E_tot

    # Plot mean connectivity history
    fig, ax = plt.subplots()
    ax.plot(cyt_conn.mean(axis=0), linewidth=2, c='b', label='Cytokine')
    ax.plot(rec_conn.mean(axis=0), linewidth=2, c='c', label='Receptor')
    ax.plot(tf_conn.mean(axis=0), linewidth=2, c='k', label='Transcription Factor')
    ax.plot(E_conn.mean(axis=0), linewidth=2, c='r', linestyle='--', label='Effector Cell')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Connectivity')
    ax.legend()
    plt.savefig(outdir + os.sep + 'mean_connectivity.pdf')

    # Save results
    np.save(outdir + os.sep + 'cyt_conn.npy', cyt_conn)
    np.save(outdir + os.sep + 'rec_conn.npy', rec_conn)
    np.save(outdir + os.sep + 'tf_conn.npy', tf_conn)


def plot_fitness_history(outdir):
    """
    Computes the average fitness per generation.
    Saves a graph of this quantity over all generations into outdir.
    """
    generations = [int(dir[1:]) for dir in os.listdir(outdir) if dir.startswith('g')]
    num_generations = max(generations)
    individuals = [int(dir[3:-4]) for dir in os.listdir(outdir + os.sep + 'g0') if dir.startswith('net')]
    num_individuals = max(individuals)

    fit_hist = np.zeros((num_individuals, num_generations))
    for g in tqdm(range(num_generations)):
        for i in tqdm(range(num_individuals), leave=False):
            ind = graph_evolution.NetworkEvolver()
            fname = os.path.join(outdir, 'g' + str(g), 'net' + str(i) + '.pkl')
            ind.load(fname)
            fit = ind.__score__()[0]
            fit_hist[i][g] = fit

    # Plot fitness history
    fig, ax = plt.subplots()
    plt.plot(fit_hist.mean(axis=0), linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Fitness')
    plt.savefig(outdir + os.sep + 'mean_fit.pdf')

    # Save fitness history
    np.save(outdir + os.sep + 'fit_hist.npy', fit_hist)


if __name__ == '__main__':
    OUTDIR = os.path.join(os.getcwd(), 'output')

    # plot_fitness_history(OUTDIR)
    # plot_mean_connectivity(OUTDIR)
    plot_perplexity(OUTDIR)
