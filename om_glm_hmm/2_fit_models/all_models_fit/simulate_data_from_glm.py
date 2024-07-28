import numpy as np
import ssm

def simulate_stim(n_trials):
    """
    n_trials: desired simulated trials
    """
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

def simulate_wsls_agents(wsls_weight_vectors,n_trials,z_stim):
    """
    wsls_weight_vectors: list of sets of weights
    n_trials: number of desired simulated trials
    z_stim: z-scored stim vector
    """
    # choice array
    y = []
    # outcome array
    outcome = []
    for i in range(n_trials):
        # initialize variables
        if i == 0:
            x_ws = 1
            x_ls = 0
            inpt_arr = np.array([z_stim[i], x_ws, x_ls, 1]).reshape(1, -1)
        # get probability
        pi = np.exp(np.dot(wsls_weight_vectors, inpt_arr[-1, :])) / (1 + np.exp(np.dot(wsls_weight_vectors, inpt_arr[-1, :])))
        # get choice from rbinom
        choice_sim = np.random.binomial(1, p=pi)
        # re-encode choice so it matches with our data
        if choice_sim == 0:
            choice_sim = -1
        # get outcome for this trial (failure or not)
        fail_sim = 1 if choice_sim * z_stim[i] > 0 else 0
        outcome_sim = 0 if choice_sim * z_stim[i] > 0 else 1
        x_ws = outcome_sim*choice_sim
        x_ls = fail_sim*choice_sim
        # get choice array
        y.append(choice_sim)
        # get outcome array
        outcome.append(outcome_sim)
        # get array for the next trial
        inpt_arr = np.vstack(
            (inpt_arr, np.array([z_stim[i + 1], x_ws, x_ls, 1]).reshape(1, -1)))
    # remove last row
    inpt_arr = inpt_arr[:-1, :]
    # add choice array
    inpt_arr = np.append(inpt_arr, np.array(y).reshape(-1, 1), axis=1)
    inpt_arr = np.append(inpt_arr, np.array(outcome).reshape(-1, 1), axis=1)
    return inpt_arr


def simulate_from_weights_pfailpchoice_model(weight_vectors,n_trials,z_stim):
    """
    weight_vectors: list of sets of weights from glmhmm states
    n_trials: number of desired simulated trials
    z-stim: z-scored stim vec
    """
    # choice array
    y = []
    # outcome array
    outcome = []
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
        outcome_sim = 0 if choice_sim*z_stim[i] > 0 else 1
        pfail = fail_sim
        pchoice = choice_sim
        # get choice array
        y.append(choice_sim)
        # get outcome array
        outcome.append(outcome_sim)
        # get array for the next trial
        inpt_arr = np.vstack((inpt_arr, np.array([pfail, z_stim[i+1], z_stim[i+1]*pfail, pchoice, 1]).reshape(1,-1)))
    # remove last row
    inpt_arr = inpt_arr[:-1,:]
    # add choice array
    inpt_arr = np.append(inpt_arr, np.array(y).reshape(-1,1),axis=1)
    inpt_arr = np.append(inpt_arr, np.array(outcome).reshape(-1, 1), axis=1)
    return inpt_arr

def simulate_from_glmhmm_pfailpchoice_model(this_hmm,n_trials,z_stim):

    # choice array
    y = []
    # for params recovery
    y_sim = []
    # state array
    state_arr = []
    # outcome array
    outcome = []
    for i in range(n_trials):
        # initialize variables
        if i == 0:
            pchoice = 1
            pfail = 1
            inpt_arr = np.array([pfail, z_stim[i], z_stim[i] * pfail, pchoice, 1]).reshape(1, -1)
            # sample data from glm-hmm model
            true_z, true_y = this_hmm.sample(T=1, input=inpt_arr[-1,:].reshape(1,-1))
            true_z_past = true_z
            true_y_past = true_y
        else:
            # with prefix for input-driven obs data simulation
            true_z, true_y = this_hmm.sample(T=1,prefix=(true_z_past,true_y_past), input=inpt_arr[-1, :].reshape(1, -1))
            true_z_past = np.append(true_z_past, true_z)
            true_y_past = np.append(true_y_past, true_y[0][0]).reshape(-1, 1)
        choice_sim = true_y[0][0]
        # re-encode choice so it matches with our data
        if choice_sim == 0:
            choice_sim = -1
        # get outcome for this trial (failure or not)
        fail_sim = 1 if choice_sim * z_stim[i] > 0 else 0
        outcome_sim = 0 if choice_sim * z_stim[i] > 0 else 1
        pfail = fail_sim
        pchoice = choice_sim
        # get state array
        state_arr.append(true_z[0])
        # get choice array
        y.append(choice_sim)
        y_sim.append(true_y[0][0])
        # get outcome array
        outcome.append(outcome_sim)
        # get array for the next trial
        inpt_arr = np.vstack(
            (inpt_arr, np.array([pfail, z_stim[i + 1], z_stim[i + 1] * pfail, pchoice, 1]).reshape(1, -1)))
        # remove last row
    inpt_arr = inpt_arr[:-1, :]
    # reshape to fit data structure assumption in ssm package
    y_sim = np.array(y_sim).reshape(-1,1)

    return inpt_arr, y_sim, y, outcome, state_arr
    # example from notebook
    # Generate a sequence of latents and choices for each session
    # true_latents, true_choices = [], []
    # for sess in range(num_sess):
    #     true_z, true_y = true_glmhmm.sample(num_trials_per_sess, input=inpts[sess])
    #     true_latents.append(true_z)
    #     true_choices.append(true_y)
    #
    # # Calculate true loglikelihood
    # true_ll = true_glmhmm.log_probability(true_choices, inputs=inpts)
    # print("true ll = " + str(true_ll))



