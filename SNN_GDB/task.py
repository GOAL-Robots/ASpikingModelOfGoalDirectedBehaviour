import numpy as np

def task_feedback(stim_acti_exp, stimulus, action, n_ttrial):
    s = None
    if stim_acti_exp[stimulus, action] == 0:
        if 0 <= n_ttrial <= 2:
            stim_acti_exp[stimulus, action] = -1
        else:
            if np.count_nonzero(stim_acti_exp == 1) == 0:
                stim_acti_exp[stimulus, action] = 1
                s = {'S1': stimulus}
            elif np.count_nonzero(stim_acti_exp == 1) == 1 \
                    and \
                    np.count_nonzero(stim_acti_exp[stimulus, :] == 1) == 0 \
                    and \
                    np.count_nonzero(stim_acti_exp[stimulus, :] == -1) == 3:
                stim_acti_exp[stimulus, action] = 1
                s = {'S2': stimulus}
            elif np.count_nonzero(stim_acti_exp == 1) == 2 \
                    and \
                    np.count_nonzero(stim_acti_exp[stimulus, :] == 1) == 0 \
                    and \
                    np.count_nonzero(stim_acti_exp[stimulus, :] == -1) == 4:
                stim_acti_exp[stimulus, action] = 1
                s = {'S3': stimulus}
            else:
                stim_acti_exp[stimulus, action] = -1
    if stim_acti_exp[stimulus, action] == 1:
        print('Yay, good job!')
        feedback = 8
    else:
        feedback = 9
    print(stim_acti_exp)
    return feedback, s