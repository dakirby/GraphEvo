import os
import immune_network
import graph_evolution
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_mean_connectivity(outdir):
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
    plot_mean_connectivity(OUTDIR)
