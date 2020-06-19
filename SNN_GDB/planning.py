import numpy as np
from SNN_GDB.functions import softmax, stoc_sele, entropy
from SNN_GDB.goal_matching import goal_matching
from SNN_GDB.focus_exp import exploration

def planning(all_inputs, chosen_colour, decay_rate, weights_inpu_asso,
             weights_asso_asso, weights_asso_outp, weights_goal_asso,
             weights_expl_acti, eligibility_trace, gamma, rl_learning_rate,
             colours, actions, feedbacks, n_ttrial, planning_trials, steps,
             items_steps, goal_change, all_asso_seq, all_outc_seq, entropy_m,
             reaction_times, temperature, noise, entropy_threshold, d_entropy):

    all_asso = np.zeros((steps - 2, weights_asso_asso.shape[1]))
    all_outp = np.zeros((steps - 2, weights_inpu_asso.shape[1]))

    _Hm = []

    for n_ptrials in range(planning_trials):
        H_m = []
        H_o = []

        # Activation vectors
        inpu_layer = np.zeros(weights_inpu_asso.shape[1])
        asso_layer = np.zeros(weights_asso_asso.shape[1])
        outp_layer = np.zeros(weights_inpu_asso.shape[1])
        goal_layer = np.zeros(weights_goal_asso.shape[1])
        decay = np.zeros(weights_asso_asso.shape[1])

        activ_story = []
        activ_story_old = []
        count_actions = np.zeros(weights_inpu_asso.shape[1])
        count_feedbacks = np.zeros(weights_inpu_asso.shape[1])

        planned_action = np.zeros(weights_inpu_asso.shape[1])
        planned_feedback = np.zeros(weights_inpu_asso.shape[1])

        for n_step in range(steps):

            # Storing old activation
            inpu_layer_old = np.copy(inpu_layer)
            asso_layer_old = np.copy(asso_layer)
            decay_old = np.copy(decay)

            if n_step == 0: asso_layer_old[-1] = 1

            # Input activation
            if n_step < items_steps[0]:
                inpu_layer = all_inputs[chosen_colour]
            else: inpu_layer = np.zeros(len(all_inputs[0]))

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
            H_mss = entropy(asso_softmax.copy())
            H_mss /= entropy(np.full(asso_softmax.shape[0],
                                     1 / asso_softmax.shape[0]))
            asso_layer = stoc_sele(asso_softmax)

            if 1 <= n_step < 16:
                H_m.append(H_mss)

            # Activation of the output / action layer
            if n_step < 2:
                outp_layer.fill(0)
            else:
                outp_activ = np.dot(weights_asso_outp, asso_layer_old)
                outp_activ += np.random.normal(0., noise, len(outp_activ))
                outp_softmax = softmax(outp_activ.copy(), temperature)
                H_out = entropy(outp_softmax.copy())
                H_out /= entropy(np.full(outp_softmax.shape[0],
                                         1 / outp_softmax.shape[0]))
                outp_layer = stoc_sele(outp_softmax)

                H_o.append(H_out)

                if np.where(outp_layer == 1)[0] in actions:
                    count_actions += np.copy(outp_layer)
                if np.where(outp_layer == 1)[0] in feedbacks:
                    count_feedbacks += np.copy(outp_layer)

            if 1 <= n_step <= 15:
                all_asso[n_step - 1, :] += asso_layer
            if n_step >= 2:
                all_outp[n_step - 2, :] += outp_layer

            activ_story.append(np.nonzero(asso_layer))
            activ_story_old.append(np.nonzero(asso_layer_old))

        entropy_plans = np.array(H_m).mean().copy()
        _Hm.append(entropy_plans)

        if entropy_plans <= entropy_threshold:
            # Plan action and feedback according to units activation
            planned_action[np.random.choice(np.where(
                count_actions[actions] == count_actions[actions].max())[0])
                           + len(colours)] = 1

            planned_feedback[np.random.choice(np.where(
                count_feedbacks[feedbacks] ==
                count_feedbacks[feedbacks].max())[0]
                                              + len(colours + actions))] = 1

            if (planned_feedback[feedbacks[0]] == 1 and goal_layer[0] == 1) \
                or \
                (planned_feedback[feedbacks[1]] == 1 and goal_layer[1] == 1):

                correct_plan = [all_inputs[chosen_colour],
                                planned_action,
                                planned_feedback]

                if entropy_plans <= entropy_threshold:
                    n_ptrials += 1
                    reaction_times[chosen_colour].append(n_ptrials)
                    entropy_m[chosen_colour].append(np.array(_Hm).mean())
                    all_asso_seq[chosen_colour].append(all_asso)
                    all_outc_seq[chosen_colour].append(all_outp)

                    return correct_plan, weights_goal_asso, entropy_m, \
                           all_asso_seq, all_outc_seq


            if (planned_feedback[feedbacks[0]] == 1 and goal_layer[1] == 1) \
                or \
                (planned_feedback[feedbacks[1]] == 1 and goal_layer[0] == 1):

                weights_goal_asso = goal_matching(goal_layer,
                                                  activ_story,
                                                  eligibility_trace,
                                                  weights_goal_asso, gamma,
                                                  rl_learning_rate)

            entropy_threshold -= d_entropy

        elif entropy_plans > entropy_threshold:
            # Colour activation
            colour_input = all_inputs[chosen_colour][:len(colours)]

            # Goal activation
            if n_ttrial < goal_change:
                goal_layer[0] = 1
            else:
                goal_layer[1] = 1

            action_exploration = exploration(colour_input,
                                             goal_layer,
                                             weights_expl_acti)

            planned_action = np.hstack((np.zeros(len(colours)),
                                        action_exploration,
                                        np.zeros(len(feedbacks))))
            if goal_layer[0] == 1:
                planned_feedback = all_inputs[feedbacks[0]]
            elif goal_layer[1] == 1:
                planned_feedback = all_inputs[feedbacks[1]]

            correct_plan = [all_inputs[chosen_colour],
                            planned_action,
                            planned_feedback]

            n_ptrials += 1
            reaction_times[chosen_colour].append(n_ptrials)
            entropy_m[chosen_colour].append(np.array(_Hm).mean())
            all_asso_seq[chosen_colour].append(all_asso)
            all_outc_seq[chosen_colour].append(all_outp)

            return correct_plan, weights_goal_asso, entropy_m, \
                   all_asso_seq, all_outc_seq
