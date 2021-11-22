import os
import immune_network
import graph_evolution
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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


if __name__ == '__main__':
    OUTDIR = os.path.join(os.getcwd(), 'output')

    plot_fitness_history(OUTDIR)