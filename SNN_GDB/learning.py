import numpy as np
from SNN_GDB.functions import softmax, stoc_sele
from SNN_GDB.goal_matching import goal_learning

def learning_online(action_feedback, decay_rate, weights_inpu_asso,
                    weights_asso_asso, weights_asso_outp, weights_goal_asso,
                    mr_learning_rate, rl_learning_rate, threshold, items_steps,
                    n_ttrial, steps, goal_change, learning_sequence,
                    temperature, noise):

    # Learning the feedback according to the task response
    if action_feedback == 8:
        actual_output = np.zeros(len(learning_sequence[2]))
        actual_output[8] = 1
    elif action_feedback == 9:
        actual_output = np.zeros(len(learning_sequence[2]))
        actual_output[9] = 1

    neuron_story = []

    # Activation vectors
    inpu_layer = np.zeros(weights_inpu_asso.shape[1])
    asso_layer = np.zeros(weights_asso_asso.shape[1])
    outp_layer = np.zeros(weights_inpu_asso.shape[1])
    goal_layer = np.zeros(weights_goal_asso.shape[1])
    decay = np.zeros(weights_asso_asso.shape[1])

    for n_step in range(steps):

        inpu_layer_old = np.copy(inpu_layer)
        asso_layer_old = np.copy(asso_layer)
        decay_old = np.copy(decay)

        if n_step == 0: asso_layer_old[-1] = 1

        # Input activation
        if n_step < items_steps[0]:
            inpu_layer = learning_sequence[0]
        elif items_steps[0] <= n_step < sum(items_steps[:2]):
            inpu_layer = learning_sequence[1]
        elif sum(items_steps[:2]) <= n_step < sum(items_steps[:3]):
            inpu_layer = actual_output

        # Goal activation
        if n_ttrial < goal_change:
            goal_layer[0] = 1
        else: goal_layer[1] = 1
        goal_activity = goal_layer + \
                        np.random.normal(0., noise, len(goal_layer))
        goal_activity[goal_activity < 0.] = 0.

        # Activation of the associative layer
        decay = (decay_rate * decay_old) + asso_layer_old
        asso_activ = np.dot(weights_inpu_asso, inpu_layer_old) + \
                     np.dot(weights_asso_asso, asso_layer_old) + \
                     np.dot(weights_goal_asso, goal_activity) - decay
        asso_activ += np.random.normal(0., noise, len(asso_layer))
        asso_activ[-1] = -10
        asso_softmax = softmax(asso_activ.copy(), temperature)
        asso_layer = stoc_sele(asso_softmax)

        # Activation of the output
        if n_step < 2:
            outp_layer.fill(0)
        else:
            if n_step < items_steps[0] + 2:
                outp_layer = learning_sequence[0]
            elif items_steps[0] + 2 <= n_step < sum(items_steps[:2]) + 2:
                outp_layer = learning_sequence[1]
            elif sum(items_steps[:2]) + 2 <= n_step < sum(items_steps[:3]) + 2:
                outp_layer = actual_output

        asso_neuron = np.where(asso_layer == 1)[0]
        outp_neuron = np.where(outp_layer == 1)[0]
        neuron_story.append(asso_neuron)

        # Learning network weights
        # Learning weights from input to associative
        dW_inpu_asso = mr_learning_rate * (np.exp(
            -weights_inpu_asso[asso_neuron, :]) * inpu_layer_old - threshold)
        weights_inpu_asso[asso_neuron] += dW_inpu_asso
        weights_inpu_asso[weights_inpu_asso <= -1] = -1

        # Learning inner associative weights
        dW_asso_asso = mr_learning_rate * (np.exp(
            -weights_asso_asso[asso_neuron, :]) * asso_layer_old - threshold)
        weights_asso_asso[asso_neuron] += dW_asso_asso
        np.fill_diagonal(weights_asso_asso, 0)
        weights_asso_asso[weights_asso_asso <= -1] = -1

        # Learning weights from associative to output
        dW_asso_outp = 0.07 * (np.exp(-weights_asso_outp[outp_neuron, :]) *
                               asso_layer_old - 0.01)
        weights_asso_outp[outp_neuron] += dW_asso_outp
        weights_asso_outp[weights_asso_outp <= -1] = -1

    goal_learning(actual_output, goal_layer, neuron_story,
                  np.full(weights_goal_asso.shape, 0.), weights_goal_asso,
                  1, rl_learning_rate)

    return weights_inpu_asso, weights_asso_asso, \
           weights_asso_outp, weights_goal_asso