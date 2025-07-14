import json
from pathlib import Path
import os
import numpy as np
import sys
import pandas as pd
sys.path.append('.')
import matplotlib.pyplot as plt
from scipy.stats import sem
# import seaborn as sns
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_global_weights, get_global_trans_mat, load_animal_list,\
    load_correct_incorrect_mat, calculate_state_permutation
from simulate_data_from_glm import *
from ssm.util import find_permutation
from scipy.optimize import curve_fit
# plot psychometrics
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

######################### PARAMS ####################################################
# NOTES:
# individual glmhmm are initialized by permuted global glmhmm weights, so there needs not be permutation params for plotting
# if matching between indiv and global glmhmm weights is desired, permutation needs to be passed when calling global glmhmm
K_max = 5
root_folder_dir = '/home/anh/Documents/phd'
root_folder_name = 'om_choice_nopfail'
root_data_dir = Path(root_folder_dir) / root_folder_name / 'data'
root_result_dir = Path(root_folder_dir) / root_folder_name / 'result'

data_dir = root_data_dir / (root_folder_name +'_data_for_cluster')
data_individual = data_dir / 'data_by_animal'
results_dir = root_result_dir / (root_folder_name +'_global_fit')
results_dir_individual = root_result_dir/ (root_folder_name + '_individual_fit')
animal_list = load_animal_list(data_individual / 'animal_list.npz')
# list of columns/predictors name as ordered by pansy
labels_for_plot = ['pfail', 'stim', 'stim_pfail', 'pchoice','bias']
# first index for all animals
n_session_lst = [1,50]              # higher n sessions (>=60) seems to yield poorer fit??
n_trials_lst = [5000,250]
cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306",
        '#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00'
    ]
#
# trouble_animals =['23.0','24.0','26.0']
# animal_list = list(set(animal_list)-set(trouble_animals))


# flag for running individual animals analysis
individual_animals = 0
exploratory_plot = 1
######################################################################################
##################### INDIVIDUAL ANIMAL ##############################################
##################### LOAD DATA ######################################################
trans_mat_all = np.ones((4,4))
if individual_animals:
    animal_lst = [str(x) for x in animal_list]
    for K in [4]:
        n_session = n_session_lst[-1]
        n_trials = n_trials_lst[-1]
        state_dwell_times = np.zeros((len(animal_list) + 1, K))
        for z,animal in enumerate(animal_list):
            print(animal)
            results_dir_individual_animal = results_dir_individual / animal
            cv_file = results_dir_individual_animal / "cvbt_folds_model.npz"
            cvbt_folds_model = load_cv_arr(cv_file)

            with open(results_dir_individual_animal / "best_init_cvbt_dict.json", 'r') as f:
                best_init_cvbt_dict = json.load(f)

            # Get the file name corresponding to the best initialization for given K value
            raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                         results_dir_individual_animal,
                                                         best_init_cvbt_dict)
            hmm_params, lls = load_glmhmm_data(raw_file)

            # Save parameters for initializing individual fits
            weight_vectors = -hmm_params[2]
            log_transition_matrix = hmm_params[1][0]
            init_state_dist = hmm_params[0][0]
            transition_matrix = np.exp(hmm_params[1][0])
            # state dwell times for individual animals
            calculate_state_permutation(hmm_params)
            state_dwell_times[z, :] = 1 / (np.ones(K) -
                                           transition_matrix.diagonal())
            # trans_mat_all = np.dstack((trans_mat_all,np.exp(log_transition_matrix)))

##############################################################################################
####################################### EXPLORATORY PLOTS ####################################
if exploratory_plot:
    inpt_data_all = pd.read_csv(os.path.join(data_dir,'om_state_info.csv'))
    index_data = inpt_data_all.index
    inpt_data_all['success_trans'] = inpt_data_all.success.copy()
    inpt_data_all.loc[inpt_data_all.success_trans == 0, 'success_trans'] = -1

    inpt_data_all['rolling_bias'] = inpt_data_all['choice_trans'].shift(periods=2)
    inpt_data_all['rolling_accuracy'] = inpt_data_all['success'].shift(periods=2)

    inpt_data_all['shited_success'] = inpt_data_all['success'].shift(periods=2)
    window_length = [5, 20, 50, 100, 150, 200, 250]
    rolling_acc_arr = np.ones((len(inpt_data_all), len(window_length)))
    for session_no in inpt_data_all.session_identifier.unique():
        session_no_index = list(index_data[(inpt_data_all['session_identifier'] == session_no)])
        inpt_data_all.loc[session_no_index, 'rolling_bias'] = inpt_data_all.loc[
            session_no_index, 'rolling_bias'].rolling(
            window=window_length[0]).mean()
        # fill the initial, engaged nan trials with median of rolling accuracy in that session
        median_sess = inpt_data_all.loc[session_no_index, 'rolling_bias'].median()
        # inplace does not work with df slices!!
        inpt_data_all.loc[session_no_index, 'rolling_bias'] = inpt_data_all.loc[
            session_no_index, 'rolling_bias'].fillna(median_sess)

        # only calculate rolling success in engaged trials
        inpt_data_all.loc[session_no_index, 'rolling_accuracy'] = inpt_data_all.loc[
            session_no_index, 'rolling_accuracy'].rolling(
            window=window_length[0]).mean()
        # fill the initial, engaged nan trials with median of rolling accuracy in that session
        median_sess = inpt_data_all.loc[session_no_index, 'rolling_accuracy'].median()
        # inplace does not work with df slices!!
        inpt_data_all.loc[session_no_index, 'rolling_accuracy'] = inpt_data_all.loc[
            session_no_index, 'rolling_accuracy'].fillna(median_sess)

        # get different rolling accuracy for different window length
        for window_id in range(len(window_length)):
            min_periods = int(window_length[window_id] / 10)
            if min_periods < 1:
                min_periods = int(min_periods * 10)
            # only calculate rolling success in engaged trials
            rolling_acc_arr[session_no_index, window_id] = inpt_data_all.loc[
                session_no_index, 'shited_success'].rolling(
                window=window_length[window_id], min_periods=min_periods).mean()
            # fill the initial, engaged nan trials with median of rolling accuracy in that session
            median_sess = np.nanmean(rolling_acc_arr[session_no_index, window_id])
            # inplace does not work with df slices!!
            rolling_acc_arr[session_no_index, window_id] = np.nan_to_num(rolling_acc_arr[session_no_index, window_id],
                                                                         nan=median_sess)

    #### check state transitions and state duration
    for K in [3,4]:
        dict_K_name = {0: 'K' + str(K) + '_state_0', 1: 'K' + str(K) + '_state_1', 2: 'K' + str(K) + '_state_2',
                       3: 'K' + str(K) + '_state_3'}
        # get state prob
        value_current_K = [v for v in dict_K_name.values()][:K]

        state_prob = np.array(inpt_data_all.loc[:, value_current_K])
        # get the values of max state prob
        states_max_posterior = np.argmax(state_prob, axis=1)
        state_prob_max = state_prob[np.arange(len(state_prob)), states_max_posterior]
        state_prob_max = np.tile(state_prob_max,[K,1]).T
        # get the difference between biggest state prob and the others
        d_state_prob = state_prob_max - state_prob
        d_state_max = np.argmax(d_state_prob,axis=1)
        d_state_prob_max = d_state_prob[np.arange(len(d_state_prob)),d_state_max]
        # check with the low prob difference, if they are usually a jump or gradual descend
        inpt_data_all['K'+str(K)+'_diff'] = d_state_prob_max
    plt.hist(inpt_data_all['K3_diff'], bins=30, alpha=0.2);
    plt.hist(inpt_data_all['K4_diff'], bins=30, alpha=0.2);
    plt.show()

    # plot diff state conditioned on pre/during/post transition point
    state_change_index = list(index_data[inpt_data_all.K3_state_change==1][1:])
    num_col = 5
    num_row = 3
    fig,ax=plt.subplots(num_row,num_col,figsize=(24,18),sharey=True)
    for sc_id in range(len(state_change_index)):
        for row_id in range(num_row):
            shift_diff_index = row_id - 1
            ax_plot = 1
            transition_diff = inpt_data_all.loc[state_change_index[sc_id]+shift_diff_index,'K3_diff']
            if transition_diff > 0.8:
                ax_plot = 4
            elif transition_diff < 0.2:
                ax_plot = 0
            elif (transition_diff <= 0.55) & (transition_diff >= 0.45):
                ax_plot= 2
            elif (transition_diff > 0.55) & (transition_diff <= 0.8):
                ax_plot = 3

            ax[row_id,ax_plot].plot(np.arange(7),inpt_data_all.loc[state_change_index[sc_id]-3:state_change_index[sc_id]+3,'K3_diff'],linestyle='--',color='grey',linewidth=0.2)
    for col_id in range(num_col):
        for row_id in range(num_row):
            ax[row_id,col_id].axvline(3,linestyle='--',color='r')
            ax[row_id,col_id].axhline(0.15, linestyle='--',color='g')
    fig.tight_layout(); fig.show()
    del fig

    # plot diff state with counts of transitioned-into states
    num_row = 2
    fig, ax = plt.subplots(num_row, num_col, figsize=(24, 12),sharey='row')
    state_lst = [[0],[0],[0],[0],[0]]
    for sc_id in range(len(state_change_index)):
        transition_diff = inpt_data_all.loc[state_change_index[sc_id] + 1, 'K3_diff']
        transition_state = inpt_data_all.loc[state_change_index[sc_id], 'K3_state']
        if transition_diff > 0.8:
            ax_plot = 4
        elif transition_diff < 0.2:
            ax_plot = 0
        elif (transition_diff <= 0.55) & (transition_diff >= 0.45):
            ax_plot = 2
        elif (transition_diff > 0.55) & (transition_diff <= 0.8):
            ax_plot = 3
        else:
            ax_plot = 1

        state_lst[ax_plot].extend([transition_state])
        ax[0, ax_plot].plot(np.arange(7),
                                 inpt_data_all.loc[state_change_index[sc_id] - 3:state_change_index[sc_id] + 3,
                                     'K3_diff'], linestyle='--', color='grey', linewidth=0.2)
    for hist_id in range(num_col):
        ax[1, hist_id].hist(state_lst[hist_id])

    for col_id in range(num_col):
        ax[0, col_id].axvline(3, linestyle='--', color='r')
    fig.tight_layout();
    fig.show()


