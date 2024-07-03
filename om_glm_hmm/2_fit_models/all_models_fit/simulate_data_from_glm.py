import numpy as np

def simulate_stim(n_trials):
    # meand of stim distribution
    mu = 0.439
    # spread of stim distribution
    std = 0.25
    # list of simulated stim
    m = []
    for i in range(n_trials):
        # get simulated stim from a normal distribution with desired mu and std
        simulated_stim = np.random.normal(mu, std, 1)[0]
        # bounded stim
        while simulated_stim < 0 or simulated_stim > 0.6:
            simulated_stim = np.random.normal(mu, std, 1)[0]
        # get sided stim by flipping a coin
        if np.random.uniform() < 0.5:
            simulated_stim = -simulated_stim
        m.append(simulated_stim)
    return m

def simulate_from_weights_pfailpchoice_model(weight_vectors,n_trials,z_stim):
    # choice array
    y = []
    for i in range(n_trials):
        # initialize variables
        if i ==0:
            pchoice = 1
            pfail = 1
            inpt_arr = np.array([pfail, z_stim[i], z_stim[i]*pfail, pchoice, 1]).reshape(1,-1)
        # get probability
        pi = np.exp(np.dot(weight_vectors,inpt_arr[-1,:])) / (1 + np.exp(np.dot(weight_vectors,inpt_arr[-1,:])))
        # get choice from rbinom
        choice_sim = np.random.binomial(1,p=pi)
        # re-encode choice so it matches with our data
        if choice_sim == 0:
            choice_sim = -1
        # get outcome for this trial (failure or not)
        fail_sim = 1 if choice_sim*z_stim[i] > 0 else 0
        pfail = fail_sim
        pchoice = choice_sim
        # get choice array
        y.append(choice_sim)
        # get array for the next trial
        inpt_arr = np.vstack((inpt_arr, np.array([pfail, z_stim[i], z_stim[i]*pfail, pchoice, 1]).reshape(1,-1)))
    return inpt_arr[:-1,:], y




