import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_context("poster", font_scale=0.6)
    # sns.set_style("whitegrid")
except:
    Warning('seaborn not installed, plots will be less cool')
    pass

from SNN_GDB.functions import partition_sum

plt.rc('font', size='10', family='consolas')


def plot_activity(all_asso_seq, stim_seq, trials=[], cmap='viridis'):
    if trials.__len__() == 0:
        trials = [all_asso_seq[0].__len__()]
    for s in stim_seq.keys():
        _as = all_asso_seq[stim_seq[s]]
        ps = partition_sum(_as, trials)

        if trials.__len__() > 1:
            fig, axes = plt.subplots(nrows=trials.__len__(), ncols=1)
            for ax, i in zip(axes.flat, ps):
                i = (i - np.min(i)) / (np.max(i) - np.min(i))  # Normalization
                im = ax.pcolormesh(i, cmap=cmap)
            fig.colorbar(im, ax=axes.ravel().tolist())
            fig.suptitle('Progressive units specialization for ' + s,
                         fontsize=20)
        else:
            plt.pcolormesh(ps[0], cmap=cmap)
            plt.colorbar()
            plt.show()
    plt.show()

    return


def plot_outcomes(all_asso_seq, stim_seq, trials=[], cmap='viridis'):
    if trials.__len__() == 0:
        trials = [all_asso_seq[0].__len__()]
    for s in stim_seq.keys():
        _as = all_asso_seq[stim_seq[s]]
        ps = partition_sum(_as, trials)

        if trials.__len__() > 1:
            fig, axes = plt.subplots(nrows=trials.__len__(), ncols=1)
            for ax, i in zip(axes.flat, ps):
                i = (i - np.min(i)) / (np.max(i) - np.min(i)) # Normalization
                im = ax.pcolormesh(i, cmap=cmap)
            fig.colorbar(im, ax=axes.ravel().tolist())
            fig.suptitle('Outcome spiking activity associated to ' + s,
                         fontsize=15)
        else:
            plt.pcolormesh(ps[0], cmap=cmap)
            plt.colorbar()
            plt.show()
    plt.show()

    return


def plot_rt_explore_exploit(rt):
    plt.plot(range(1, len(rt) + 1), rt, marker='o')
    plt.suptitle('Avg RT', fontsize=15)
    plt.title('(trials grouped by exploration/exploitation)')
    plt.show()

    return


def plot_sr_times(r_times):
    plt.plot(range(1, 41), r_times[0], '-o', color='b', label='S1 RT')
    plt.plot(range(1, 41), r_times[1], '-o', color='r', label='S2 RT')
    plt.plot(range(1, 41), r_times[2], '-o', color='g', label='S3 RT')
    plt.plot(range(1, 41), r_times.mean(0), '--', color='k',
             label='Average RT')
    plt.title('Single stimulus RT on trial')
    plt.legend()
    plt.show()

    return
