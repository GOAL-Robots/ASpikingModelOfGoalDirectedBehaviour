import numpy as np
import sys
import os
import scipy.special as ss

def gen_inputs(n_of_units, items):
    all_inputs = []
    for i in items:
        _inpu = np.zeros(n_of_units)
        _inpu[i] = 1
        all_inputs.append(_inpu)
    return all_inputs


def softmax(x, T=0.01):
    # Softmax function algorithm for stochastics selection
    # T value defines the temperature
    x -= ss.logsumexp(x)
    return np.exp(x / T) / np.sum(np.exp(x / T))


def stoc_sele(j):
    # Algorithm for the stochastic selection based on the softmax values
    return np.random.multinomial(1, j)


def spikes_counter(activ_steps, activ_story):
    activ_story.pop(0)
    activ_story.pop(-1)
    for a in range(len(activ_story)):
        activ_steps[a, activ_story[a]] += 1
    return activ_steps


def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    return


def entropy(v):
    v += 1e-4
    pA = v / v.sum()
    return -np.sum(pA * np.log2(pA))


def partition_sum(list, indices):
    slice = []
    idx1 = [0] + indices
    del idx1[-1]
    idx2 = indices
    for i, j in zip(idx1, idx2):
        slice.append(sum(list[i:j]))
    return slice
