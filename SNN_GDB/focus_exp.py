import numpy as np
from SNN_GDB.functions import softmax, stoc_sele

def exploration(colour_input, goal_layer, weights_expl_acti):

    inpu_goal_expl = np.dot(np.expand_dims(colour_input, 1),
                            np.expand_dims(goal_layer, 0)).flatten()

    temperature = 0.015

    acti_activ = np.dot(weights_expl_acti, inpu_goal_expl)
    acti_softmax = softmax(acti_activ, temperature)
    acti_layer = stoc_sele(acti_softmax)

    return acti_layer


def learn_expl(n_ttrial, goal_change, colours, actions, chosen_colour,
               n_goal_unit, action, action_feedback, weights_expl_acti):

    colour_input = np.zeros(len(colours))
    colour_input[chosen_colour] = 1

    action_output = np.zeros(len(actions))
    action_output[action] = 1

    goal_layer  = np.zeros(n_goal_unit)
    if n_ttrial < goal_change:
        goal_layer[0] = 1
    else: goal_layer[1] = 1

    if (action_feedback == 8 and goal_layer[0] == 1) or \
            (action_feedback == 9 and goal_layer[1] == 1):
        reward = 1
        rl_learning_rate = 0.005
    else:
        reward = -1
        rl_learning_rate = 0.5

    inpu_goal_expl = np.dot(np.expand_dims(colour_input, 1),
                            np.expand_dims(goal_layer, 0)).flatten()
    eligibility_trace = np.dot(np.expand_dims(action_output, 1),
                               np.expand_dims(inpu_goal_expl, 0))
    weights_expl_acti += rl_learning_rate * reward * eligibility_trace * \
                         ((1.5 - abs(weights_expl_acti)) / 1.5)

    return weights_expl_acti
