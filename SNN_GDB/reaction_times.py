import numpy as np

def events_labeling(events, reaction_times, mode='action'):
    # Transforming data
    events[events == 8] = 1
    events[events == 9] = 0
    events = events.astype(int)
    reaction_times = np.array((reaction_times))

    # Create SAR
    s = events[0].copy()
    a = events[1].copy()
    r = events[2].copy()

    # Create Labels for representative learning trials
    sar = np.zeros((len(s), 5))
    Q = np.zeros((3, 5))    # Association matrix of tried responses
    R = np.zeros(5)         # Association matrix of rewards
    L = np.zeros(len(s))    # Action-based Labels
    M = np.zeros(len(s))    # Reward-based Labels
    times = np.zeros((len(s))) # Event-associated times


    for i in range(len(s)):
        # Create action-based labels for learning
        Q[s[i], a[i]] = 1

        # Number of tried responses nTR
        nTR = np.sum(Q, axis=1)

        # Search phase: if the association has not been found,
        # the label is the number of tried reponses
        if R[s[i]] == 0:
            L[i] = nTR[s[i]]

        # Consolidation phase: if the association is found,
        # the label is the number of correct responses + 5
        if R[s[i]] > 0:
            L[i] = R[s[i]] + 5

        # Update Reward matrix
        R[s[i]] = r[i] + R[s[i]]

        # Create reward-based labels for learning
        # Search phase: if the association has not been found,
        # the label is the number of tried reponses
        if R[s[i]] == 0:
            M[i] = nTR[s[i]]

        # Consolidation phase: if the association is found,
        # the label is the number of correct rewards + 4
        if R[s[i]] > 0:
            M[i] = R[s[i]] + 4

        # Crate SAR + Learning Labels
        sar[i, :] = [s[i]+1, a[i]+1, r[i], L[i], M[i]]

    t0, t1, t2 = 0, 0, 0
    for j, i in zip(s, range(len(s))):
        if j == 0:
            times[i] = reaction_times[j, t0]
            t0 += 1
        elif j == 1:
            times[i] = reaction_times[j, t1]
            t1 += 1
        elif j == 2:
            times[i] = reaction_times[j, t2]
            t2 += 1

    if mode == 'action':
        return L, times
    elif mode == 'reward':
        return M, times

def label_based_rt(events, reaction_times):
    labels, times = events_labeling(events, reaction_times, mode='action')

    r_t = np.zeros(int(np.max(labels)))
    for i in range(1, int(np.max(labels)) + 1):
        _srt = np.where(labels == i)
        r_t[i - 1] = times[_srt].mean()

    return r_t
