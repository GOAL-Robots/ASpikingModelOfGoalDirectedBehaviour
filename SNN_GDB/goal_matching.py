import numpy as np

def goal_matching(goal_layer, activ_story, eligibility_trace,
                  weights_goal_asso, gamma, rl_learning_rate):

    reward = -1

    c_s = np.nonzero(goal_layer)
    for a_s in activ_story[1:-1]:
        eligibility_trace_old = eligibility_trace
        eligibility_trace = gamma * eligibility_trace_old
        eligibility_trace[a_s, c_s] += 1.

    weights_goal_asso += rl_learning_rate * reward * eligibility_trace * \
                         ((.5 - abs(weights_goal_asso)) / .5)

    return weights_goal_asso


def goal_learning(planned_feedback, goal_layer, activ_story, eligibility_trace,
                  weights_goal_asso, gamma, rl_learning_rate):

    outc_goal_match = np.full(goal_layer.shape, 0)
    reward = 1

    if planned_feedback[8] == 1:
        outc_goal_match[0] = 1
    elif planned_feedback[9] == 1:
        outc_goal_match[1] = 1

    c_s = np.nonzero(outc_goal_match)
    for a_s in activ_story[1:-1]:
        eligibility_trace_old = eligibility_trace
        eligibility_trace = gamma * eligibility_trace_old
        eligibility_trace[a_s, c_s] += 1.
    weights_goal_asso += rl_learning_rate * reward * eligibility_trace * \
                         ((.5 - abs(weights_goal_asso)) / .5)

    return weights_goal_asso
