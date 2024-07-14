import json
from pathlib import Path
import os
import numpy as np
import sys
import pandas as pd
sys.path.append('.')
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct
from simulate_data_from_glm import *
from ssm.util import find_permutation
def load_cv_arr(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    cvbt_folds_model = data[0]
    return cvbt_folds_model

def load_glmhmm_data(data_file):
    container = np.load(data_file, allow_pickle=True)
    data = [container[key] for key in container]
    this_hmm_params = data[0]
    lls = data[1]
    return [this_hmm_params, lls]

def get_file_name_for_best_model_fold(cvbt_folds_model, K, overall_dir,
                                      best_init_cvbt_dict):
    '''
    Get the file name for the best initialization for the K value specified
    :param cvbt_folds_model:
    :param K:
    :param models:
    :param overall_dir:
    :param best_init_cvbt_dict:
    :return:
    '''
    # Identify best fold for best model:
    # loc_best = K - 1
    loc_best = 0
    best_fold = np.where(cvbt_folds_model[loc_best, :] == max(cvbt_folds_model[
                                                              loc_best, :]))[
        0][0]
    base_path = overall_dir / ('GLM_HMM_K_' + str(K)) / ('fold_' + str(
        best_fold))
    key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(best_fold)
    best_iter = best_init_cvbt_dict[key_for_dict]
    raw_file = base_path / ('iter_' + str(
        best_iter)) / ('glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz')
    return raw_file

##################### PARAMS
K_max = 5
root_folder_dir = '/home/anh/Documents'
root_folder_name = 'om_choice'
root_data_dir = Path(root_folder_dir) / root_folder_name / 'data'
root_result_dir = Path(root_folder_dir) / root_folder_name / 'result'

data_dir = root_data_dir / (root_folder_name +'_data_for_cluster')
data_individual = data_dir / 'data_by_animal'
results_dir = root_result_dir / (root_folder_name +'_global_fit')
results_dir_individual = root_result_dir/ (root_folder_name + '_individual_fit')

labels_for_plot = ['pfail', 'stim', 'stim_pfail', 'pchoice','bias']
######################################################################################
##################### ALL ANIMALS ####################################################
##################### LOAD DATA ######################################################
# load dictionary for best cv model run
with open(results_dir / "best_init_cvbt_dict.json", 'r') as f:
    best_init_cvbt_dict = json.load(f)
# load cv array
cv_file = results_dir / 'cvbt_folds_model.npz'
cvbt_folds_model = load_cv_arr(cv_file)

# get file name with best initialization given K value
K=4
raw_file = get_file_name_for_best_model_fold(
    cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
hmm_params, lls = load_glmhmm_data(raw_file)
weight_vectors = -hmm_params[2]
log_transition_matrix = hmm_params[1][0]
init_state_dist = hmm_params[0][0]
# load animal data for simulation
inpt, y, session = load_data(data_dir / 'all_animals_concat.npz')
inpt_unnorm, _, _ = load_data(data_dir / 'all_animals_concat_unnormalized.npz')
# create dataframe all animals for plotting
inpt_unnorm = np.append(inpt_unnorm, np.ones(inpt_unnorm.shape[0]).reshape(-1,1),axis=1)
inpt_data = pd.DataFrame(data=inpt_unnorm,columns=labels_for_plot)
inpt_data['choice'] = y
# prepare data
violation_idx = np.where(y == -1)[0]
nonviolation_idx, mask = create_violation_mask(violation_idx,
                                               inpt.shape[0])
y[np.where(y == -1), :] = 1
inputs, datas, train_masks = partition_data_by_session(
    np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)
M = inputs[0].shape[1]
D = datas[0].shape[1]
# get posterior probs for state inference
posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                         hmm_params, K, range(K))
states_max_posterior = np.argmax(posterior_probs, axis=1)
inpt_data['state'] = states_max_posterior
##################### SIMULATE VEC #################################################
# SIMULATE FROM GLMHMM
# define col names for simulated dataframe
col_names_glmhmm = labels_for_plot + ['choice','outcome','stim_org','state', 'session','y']
n_session = 5
n_trials = 250
# instantiate model
data_hmm = ssm.HMM(K,
                       D,
                       M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=2),
                       transitions="standard")
data_hmm.params = hmm_params
# initialize list of inputs and y
glmhmm_inpt_lst, glmhmm_y_lst =[], []
# initialize simulated array
glmhmm_inpt_arr = np.zeros(len(col_names_glmhmm)).reshape(1,-1)
for i in range(n_session):
    # simulate stim vec
    stim_vec_sim = simulate_stim(n_trials + 1)
    # z score stim vec
    z_stim_sim = (stim_vec_sim - np.mean(stim_vec_sim)) / np.std(stim_vec_sim)
    # simulate data
    glmhmm_inpt, glmhmm_y, glmhmm_choice, glmhmm_outcome, glmhmm_state_arr = simulate_from_glmhmm_pfailpchoice_model(data_hmm,n_trials,z_stim_sim)
    # append list for model fit and recovery
    glmhmm_inpt_lst.append(glmhmm_inpt)
    glmhmm_y_lst.append(glmhmm_y)
    # append array for plotting
    glmhmm_inpt = np.append(glmhmm_inpt, np.array(glmhmm_choice).reshape(-1, 1), axis=1)
    glmhmm_inpt = np.append(glmhmm_inpt, np.array(glmhmm_outcome).reshape(-1, 1), axis=1)
    glmhmm_inpt = np.append(glmhmm_inpt, np.array(stim_vec_sim[:-1]).reshape(-1, 1), axis=1)
    # add state info
    glmhmm_inpt = np.append(glmhmm_inpt, np.array(glmhmm_state_arr).reshape(-1, 1), axis=1)
    # add session info
    glmhmm_inpt = np.append(glmhmm_inpt, i * np.ones(glmhmm_inpt.shape[0]).reshape(-1, 1), axis=1)
    # add y
    glmhmm_inpt = np.append(glmhmm_inpt, np.array(glmhmm_y).reshape(-1, 1), axis=1)
    # stack array
    glmhmm_inpt_arr = np.vstack((glmhmm_inpt_arr,glmhmm_inpt))
# create dataframe for plotting
glmhmm_sim_df = pd.DataFrame(data=glmhmm_inpt_arr[1:,:], columns=col_names_glmhmm)
# Calculate true loglikelihood
true_ll = data_hmm.log_probability(glmhmm_y_lst, inputs=glmhmm_inpt_lst)
print("true ll = " + str(true_ll))
# fit glmhmm and perform recovery analysis
recovered_glmhmm = ssm.HMM(K, D, M, observations="input_driven_obs",
                   observation_kwargs=dict(C=2), transitions="standard")
N_iters = 200 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
fit_ll = recovered_glmhmm.fit(glmhmm_y_lst, inputs=glmhmm_inpt_lst, method="em", num_iters=N_iters, tolerance=10**-4)
# permute states
# have to use the whole array beecause not all sessions have 4 states
# recovered_glmhmm.permute(find_permutation(np.array(glmhmm_sim_df.state), recovered_glmhmm.most_likely_states(np.array(glmhmm_sim_df.y), input=np.array(glmhmm_sim_df[labels_for_plot]))))
recovered_glmhmm.permute(find_permutation(np.array(glmhmm_state_arr), recovered_glmhmm.most_likely_states(glmhmm_y_lst[-1], input=glmhmm_inpt_lst[-1])))

# CHECK PLOTS FOR SIMULATION AND RECOVERY
# Plot the log probabilities of the true and fit models. Fit model final LL should be greater than or equal to true LL.
fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(fit_ll, label="EM")
plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ':k', label="True")
plt.legend(loc="lower right")
plt.xlabel("EM Iteration")
plt.xlim(0, len(fit_ll))
plt.ylabel("Log Probability")
plt.show()

# plot generative and recovered transitrion matrix
num_states=K
fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
gen_trans_mat = np.exp(log_transition_matrix)
plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
for i in range(gen_trans_mat.shape[0]):
    for j in range(gen_trans_mat.shape[1]):
        text = plt.text(j, i, str(np.around(gen_trans_mat[i, j], decimals=2)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.ylim(num_states - 0.5, -0.5)
plt.ylabel("state t", fontsize = 15)
plt.xlabel("state t+1", fontsize = 15)
plt.title("generative", fontsize = 15)

plt.subplot(1, 2, 2)
recovered_trans_mat = np.exp(recovered_glmhmm.transitions.log_Ps)
plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
for i in range(recovered_trans_mat.shape[0]):
    for j in range(recovered_trans_mat.shape[1]):
        text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=2)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
plt.ylim(num_states - 0.5, -0.5)
plt.title("recovered", fontsize = 15)
plt.subplots_adjust(0, 0, 1, 1);plt.tight_layout();plt.show()

##################### PSYCHOMETRIC CURVES ##########################################
# since min/max freq_trans is -1.5/1.5
bin_lst = np.arange(-1.55,1.6,0.1)
bin_name=np.round(np.arange(-1.5,1.6,.1),2)
# get binned freqs for psychometrics for simulated data
glmhmm_sim_df["binned_freq"] = pd.cut(glmhmm_sim_df['stim_org'], bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)
sim_stack = glmhmm_sim_df.groupby(['binned_freq','state'])['choice'].value_counts(normalize=True).unstack('choice').reset_index()
sim_stack[-1] = sim_stack[-1].fillna(0)
# get binned freqs for psychometrics for real data
inpt_data["binned_freq"] = pd.cut(inpt_data['stim'], bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)
data_stack = inpt_data.groupby(['binned_freq','state'])['choice'].value_counts(normalize=True).unstack('choice').reset_index()
data_stack[0] = data_stack[0].fillna(0)
data_stack['binned_freq'] = data_stack['binned_freq'].astype('float')
sns.lineplot(sim_stack.binned_freq,sim_stack[-1],hue=sim_stack.state);plt.show()

##################### PLOT WEIGHTS FOR EACH K ######################################
fig, ax= plt.subplots(2,K,figsize=(10,6),sharey="row",sharex='row')
for ax_ind in range(K):
    # plot k-state weights
    ax[0,ax_ind].plot(labels_for_plot,np.squeeze(weight_vectors[ax_ind,:,:]))
    ax[0,ax_ind].set_xticklabels(labels_for_plot, fontsize=12,rotation=45)
    ax[0,ax_ind].axhline(0,linewidth=0.5,linestyle='--')
    ax[0,ax_ind].set_title('state '+ str(ax_ind))
    # plot psychometrics for simulated data
    ax[1,ax_ind].plot(sim_stack.binned_freq.unique(),sim_stack[-1].loc[sim_stack.state==ax_ind])
    # plot psychometrics for aggregated real data
    ax[1, ax_ind].plot(sim_stack.binned_freq.unique(), data_stack[0].loc[(data_stack.state == ax_ind)
                                                                        &(data_stack.binned_freq>-0.7)
                                                                        &(data_stack.binned_freq<0.7)],'--')
    # miscellaneous
    ax[1, ax_ind].set_xticklabels(labels= [str(x) for x in sim_stack.binned_freq.unique()] ,fontsize=12, rotation=45)
    ax[1, ax_ind].axhline(0.5, linewidth=0.5, linestyle='--')
    ax[1, ax_ind].axvline(6, linewidth=0.5, linestyle='--')
plt.tight_layout()
plt.show()

##################### TRANSITION MATRIX ############################################
transition_matrix = np.exp(log_transition_matrix)
fig = plt.figure(figsize=(3, 3))
plt.subplots_adjust(left=0.3, bottom=0.3, right=0.95, top=0.95)
plt.imshow(transition_matrix, vmin=-0.8, vmax=1, cmap='bone')
for i in range(transition_matrix.shape[0]):
    for j in range(transition_matrix.shape[1]):
        text = plt.text(j,
                        i,
                        str(np.around(transition_matrix[i, j],
                                      decimals=2)),
                        ha="center",
                        va="center",
                        color="k",
                        fontsize=10)
plt.xlim(-0.5, K - 0.5)
plt.xticks(range(0, K),
           ('1', '2', '3', '4', '4', '5', '6', '7', '8', '9', '10')[:K],
           fontsize=10)
plt.yticks(range(0, K),
           ('1', '2', '3', '4', '4', '5', '6', '7', '8', '9', '10')[:K],
           fontsize=10)
plt.ylim(K - 0.5, -0.5)
plt.ylabel("state t-1", fontsize=10)
plt.xlabel("state t", fontsize=10)

##################### INDIVIDUAL ANIMAL ##############################################
##################### LOAD DATA ######################################################
animal='17.0'
results_dir_individual_animal = results_dir_individual / animal
cv_file = results_dir_individual_animal / "cvbt_folds_model.npz"
cvbt_folds_model = load_cv_arr(cv_file)

with open(results_dir_individual_animal / "best_init_cvbt_dict.json", 'r') as f:
    best_init_cvbt_dict = json.load(f)

# Get the file name corresponding to the best initialization for given K
# value
raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                             results_dir_individual_animal,
                                             best_init_cvbt_dict)
hmm_params, lls = load_glmhmm_data(raw_file)

# Save parameters for initializing individual fits
weight_vectors = -hmm_params[2]
log_transition_matrix = hmm_params[1][0]
init_state_dist = hmm_params[0][0]
# Also get data for animal:
inpt, y, session = load_data(data_individual / (animal + '_processed.npz'))
all_sessions = np.unique(session)

##################### PLOT POSTERIOR PROBS (ANIMAL SPECIFIC)

# Create mask:
# Identify violations for exclusion:
violation_idx = np.where(y == -1)[0]
nonviolation_idx, mask = create_violation_mask(violation_idx,
                                               inpt.shape[0])
y[np.where(y == -1), :] = 1
inputs, datas, train_masks = partition_data_by_session(
    np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask,
    session)

posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                         hmm_params, K, range(K))
states_max_posterior = np.argmax(posterior_probs, axis=1)

sess_to_plot_all = [all_sessions[5],all_sessions[15],all_sessions[25],all_sessions[30],all_sessions[10]]
sess_to_plot = sess_to_plot_all[:K]
cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00']
fig,ax = plt.subplots(2,K,figsize=(10, 4),sharey='row')
plt.subplots_adjust(wspace=0.2, hspace=0.9)
for i, sess in enumerate(sess_to_plot):
    # plt.subplot(2, K, i+1)
    ax[0,i].plot(labels_for_plot, np.squeeze(weight_vectors[i, :, :]))
    ax[0,i].set_xticklabels(labels_for_plot, fontsize=12, rotation=45)
    ax[0,i].axhline(0, linewidth=0.5, linestyle='--')

for i, sess in enumerate(sess_to_plot):
    # plt.subplot(2, K, i + K+1)
    # get session index from session array
    idx_session = np.where(session == sess)
    # get input according to the session index
    this_inpt = inpt[idx_session[0], :]
    # get posterior probs according to session index
    posterior_probs_this_session = posterior_probs[idx_session[0], :]
    # Plot trial structure for this session too:
    for k in range(K):
        ax[1,i].plot(posterior_probs_this_session[:, k],
                 label="State " + str(k + 1), lw=1,
                 color=cols[k])
    # get max probs of state of each trial
    states_this_sess = states_max_posterior[idx_session[0]]
    # get state change index
    state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
    # plot state change
    for change_loc in state_change_locs:
        ax[1,i].axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
    plt.ylim((-0.01, 1.01))
    plt.title("example session " + str(i + 1), fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if i == 0:
        plt.xlabel("trial #", fontsize=10)
        plt.ylabel("p(state)", fontsize=10)
    if i == len(sess_to_plot)-1:
        plt.legend(loc='upper right', frameon=False)
    else:
        plt.legend('')
plt.show()


# TODO:
# compare simulated and real data, see what the model captures in the data and what's lacking
# understand the state change dynamics
# get psychometrics for each state and fraction correct, and fraction occupation
## figure 5def in the papser
## get posterior prob for each trial, get the psychometric accordingly and compare to simulated data from model each k
# systematically plot these diagnostic plots to understand the states
#