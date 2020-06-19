"""
Created on Wed Aug 02 11:49:28 2017

@author: Ruggero Basanisi
"""

##############################################################################
from __future__ import division
import sys
import numpy as np
import copy
from SNN_GDB.functions import gen_inputs
from SNN_GDB.planning import planning
from SNN_GDB.task import task_feedback
from SNN_GDB.learning import learning_online
from SNN_GDB.focus_exp import learn_expl
from SNN_GDB.reaction_times import label_based_rt
from SNN_GDB.plots import (plot_activity,
                           plot_outcomes,
                           plot_sr_times,
                           plot_rt_explore_exploit)

###############################################################################
# Here you can set a seed for the random generator in order to reproduce datas
Ver = 'ver_0.7.6'
# seed = 23
# np.random.seed(seed)

###############################################################################

# DATA STRUCTURE

# Varibles Setting
n_asso_unit = 400   # Associative units, min 215 for this task
n_inpu_unit = 10    # Input units
n_goal_unit = 2     # Goal units

# Number of trial ad steps per trial
task_trials = 120
goal_change = 60 # Number of trials before the goal of the task changes from
                 # 'obtain all positive outcomes' to 'obtain all negative
                 # outcomes'

# Defining network steps
planning_trials = 20    # n. of planning trials per task trial
steps = 17              # n. of steps per planning/learning trial
items_steps = [5, 5, 5]
assert sum(items_steps) + 2 == steps, 'No coincidence between total steps ' \
                                      'and singular items steps'

# Units' action potential refractory time rate
decay_speed = 1
decay_rate = 0.9

# Learning parameters for the associative layer
mr_learning_rate = 0.907686308511564
threshold =  0.66747311576003

# Learning parameters for the goal layer
gamma = 1.
rl_learning_rate = 0.007601322993333

# Number of inputs by type
n_colours = 3
n_actions = 5
n_feedbacks = 2

temp = 0.0204839610927673
g_noise = 0.0201046916532638
entr_threshold = 0.736620860124126
dec_entropy = 0.11595092876422

###############################################################################
def network_start(n_asso_unit, n_inpu_unit, n_goal_unit, task_trials,
                  goal_change, planning_trials, steps, items_steps,
                  mr_learning_rate, threshold, gamma, rl_learning_rate,
                  n_colours, n_actions, n_feedbacks, temperature,
                  noise, entropy_threshold, d_entropy):

    entropy_m = [[], [], []]
    reaction_times = [[], [], []]

    all_colours = [0, 1, 2]
    all_actions = [3, 4, 5, 6, 7]
    all_feedbacks = [8, 9]
    colours = all_colours[:n_colours]
    actions = all_actions[:n_actions]
    feedbacks = all_feedbacks[:n_feedbacks]

    all_inputs = gen_inputs(n_inpu_unit, colours + actions + feedbacks)
    stim_acti_exp = np.zeros((n_colours, n_actions))

    trial_events = []
    stim_sequence = {}

    # Weight's matrices
    weights_inpu_asso = np.random.uniform(-0.05, 0.05,
                                          (n_asso_unit, n_inpu_unit))
    weights_asso_asso = np.random.uniform(-0.05, 0.05,
                                          (n_asso_unit , n_asso_unit))
    weights_asso_outp = np.copy(weights_inpu_asso.T)
    weights_goal_asso = np.random.uniform(-0., 0.,
                                          (n_asso_unit, n_goal_unit))
    weights_expl_acti = np.random.uniform(-0.00, 0.00,
                                          (n_actions,
                                           (n_colours * n_goal_unit)))


    eligibility_trace = np.copy(weights_goal_asso)

    possible_colours = []
    all_asso_seq = [[], [], []]
    all_outc_seq = [[], [], []]
    events = np.zeros((3, task_trials))

    for n_ttrial in range(task_trials):
        print('\nTial number', n_ttrial)

        # The task decide the colour
        if len(possible_colours) == 0:
            possible_colours = copy.copy(colours)
        chosen_colour = possible_colours.pop(
            np.random.randint(len(possible_colours)))

        planning_phase = planning(all_inputs, chosen_colour, decay_rate,
                                  weights_inpu_asso, weights_asso_asso,
                                  weights_asso_outp, weights_goal_asso,
                                  weights_expl_acti, eligibility_trace, gamma,
                                  rl_learning_rate, colours, actions,
                                  feedbacks, n_ttrial, planning_trials, steps,
                                  items_steps, goal_change, all_asso_seq,
                                  all_outc_seq, entropy_m, reaction_times,
                                  temperature, noise, entropy_threshold,
                                  d_entropy)

        entropy_m = planning_phase[2]

        # Executing action in the task
        stimulus = chosen_colour
        action = np.nonzero(planning_phase[0][1])[0][0] - len(colours)
        learning_sequence = planning_phase[0]
        task_output = task_feedback(stim_acti_exp, stimulus, action, n_ttrial)
        action_feedback = task_output[0]

        events[0, n_ttrial] = stimulus
        events[1, n_ttrial] = action
        events[2, n_ttrial] = action_feedback

        if task_output[1] != None:
            stim_sequence.update(task_output[1])

        learn_expl(n_ttrial, goal_change, colours, actions, chosen_colour,
                   n_goal_unit, action, action_feedback, weights_expl_acti)

        learning_online(action_feedback, decay_rate, weights_inpu_asso,
                        weights_asso_asso, weights_asso_outp,
                        weights_goal_asso, mr_learning_rate, rl_learning_rate,
                        threshold, items_steps, n_ttrial, steps,
                        goal_change, learning_sequence, temperature, noise)

        trial_events.append([chosen_colour, action, action_feedback])

    sub_rt = [reaction_times[i][:20] for i in range(len(reaction_times))]
    sub_ev = np.array([events[i][:60] for i in range(len(events))])
    lrt = label_based_rt(sub_ev, sub_rt)
    avg_rt = lrt

    if len(stim_sequence) < 3:
        s_reac_times = None
    else:
        s_reac_times = np.vstack((reaction_times[stim_sequence['S1']],
                                  reaction_times[stim_sequence['S2']],
                                  reaction_times[stim_sequence['S3']]))

    if 'joblib' not in sys.modules:
        plot_rt_explore_exploit(lrt)

        plot_sr_times(s_reac_times)


        plot_activity(all_asso_seq, stim_sequence, [3, 15, 20],
                      cmap='gist_heat_r')
        plot_outcomes(all_outc_seq, stim_sequence, [3, 15, 20],
                      cmap='viridis')

    return weights_inpu_asso, weights_asso_asso, weights_asso_outp, \
           weights_goal_asso, trial_events, str(stim_sequence), \
           avg_rt, s_reac_times


if __name__ == '__main__':
    network_start(n_asso_unit, n_inpu_unit, n_goal_unit, task_trials,
                  goal_change, planning_trials, steps, items_steps,
                  mr_learning_rate, threshold, gamma, rl_learning_rate,
                  n_colours, n_actions, n_feedbacks, temp, g_noise,
                  entr_threshold, dec_entropy)
