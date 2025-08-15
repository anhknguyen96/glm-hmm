import json
from pathlib import Path
import os
import numpy as np
import sys
import pandas as pd
from collections import Counter
# sys.path.append('.')
import matplotlib.pyplot as plt
# from scipy.stats import sem
# import seaborn as sns
# from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
#     get_file_name_for_best_model_fold, partition_data_by_session, \
#     create_violation_mask, get_marginal_posterior, get_global_weights, get_global_trans_mat, load_animal_list,\
#     load_correct_incorrect_mat, calculate_state_permutation
# from simulate_data_from_glm import *
# from ssm.util import find_permutation
# from scipy.optimize import curve_fit
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
# animal_list = load_animal_list(data_individual / 'animal_list.npz')
save_folder = Path(root_folder_dir) / root_folder_name / 'plots'
# list of columns/predictors name as ordered by pansy
labels_for_plot = ['pfail', 'stim', 'stim_pfail', 'pchoice','bias']
# first index for all animals
n_session_lst = [1,50]              # higher n sessions (>=60) seems to yield poorer fit??
n_trials_lst = [5000,250]
cols = [ "#15b01a", "#f97306", "#7e1e9c", "#3498db"
    ]
# trouble_animals =['23.0','24.0','26.0']
# animal_list = list(set(animal_list)-set(trouble_animals))
get_opto = 0

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
    if get_opto:
        opto_name = 'opto_'
    else:
        opto_name = ''

    inpt_data_all = pd.read_csv(os.path.join(data_dir,opto_name + 'om_state_info.csv'))
    index_data = inpt_data_all.index
    # now mark session start
    inpt_data_all['session'] = inpt_data_all['session_identifier'].str.split('s', expand=True)[1].astype(float)
    inpt_data_all['session_start'] = inpt_data_all.session.diff()
    ## get state probability differences
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

        # get indices of state change
        state_change_index = list(index_data[inpt_data_all['K' + str(K) + '_state_change'] == 1])
        # get the difference between state changes (hence duration)
        state_change_index_diff = np.diff(state_change_index)
        # add the duration of the last state change
        state_change_index_diff = np.append(state_change_index_diff, len(inpt_data_all) - state_change_index[-1])

        # create state dur column
        state_dur_arr = np.zeros(len(inpt_data_all))
        # allocate state dur values at every state change
        np.put(state_dur_arr, state_change_index, state_change_index_diff)
        inpt_data_all['K' + str(K) + '_state_dur'] = state_dur_arr

        # copy so it doesnt change the original values
        state_processed = inpt_data_all['K' + str(K) + '_state'].copy().values
        state_bef_aft = pd.DataFrame()
        state_dur_lst = [1,2,3,4]
        for state_dur_id in range(len(state_dur_lst)):
            if state_dur_lst[state_dur_id] == 4:
                state_change_index = list(index_data[(inpt_data_all['K' + str(K) + '_state_change'] == 1) &
                                                     (inpt_data_all['K' + str(K) + '_state_dur'] >= state_dur_lst[state_dur_id])])[1:]
            else:
                state_change_index = list(index_data[(inpt_data_all['K' + str(K) + '_state_change'] == 1) &
                                                     (inpt_data_all['K' + str(K) + '_state_dur'] == state_dur_lst[state_dur_id])])

            state_bef_index = np.array(state_change_index) - 1
            state_aft_index = np.array(state_change_index) + state_dur_id + 1

            # now update K3 state - the switching state issue remains if the state switching are close to each other
            # however the number of short states reduced significantly - from 2/3 digits to 1
            if state_dur_lst[state_dur_id] != 4:
                np.put(state_processed, state_change_index, inpt_data_all['K' + str(K) + '_state'].iloc[state_bef_index].values)
                print(np.unique(state_processed[state_change_index] == inpt_data_all['K' + str(K) + '_state'].iloc[state_bef_index].values))
                if state_dur_lst[state_dur_id] > 1 and state_dur_lst[state_dur_id] <=3:
                    np.put(state_processed, np.array(state_change_index) + 1,
                           inpt_data_all['K' + str(K) + '_state'].iloc[state_bef_index].values)

                    print(np.unique(
                        state_processed[np.array(state_change_index) + 1] == inpt_data_all['K' + str(K) + '_state'].iloc[state_bef_index].values))
                    if state_dur_lst[state_dur_id] == 3:
                        np.put(state_processed, np.array(state_change_index) + 2,
                                   inpt_data_all['K' + str(K) + '_state'].iloc[state_bef_index].values)

                        print(np.unique(
                            state_processed[np.array(state_change_index) + 2] == inpt_data_all['K' + str(K) + '_state'].iloc[
                                state_bef_index].values))

            state_curr = inpt_data_all['K' + str(K) + '_state'].iloc[state_change_index].values
            # state_processed_arr = inpt_data_all['K3_state_processed'].iloc[state_change_index].values
            state_bef = inpt_data_all['K' + str(K) + '_state'].iloc[state_bef_index].values
            state_aft = inpt_data_all['K' + str(K) + '_state'].iloc[state_aft_index].values
            state_dur = inpt_data_all['K' + str(K) + '_state_dur'].iloc[state_change_index].values
            state_diff = inpt_data_all['K' + str(K) + '_diff'].iloc[state_change_index].values

            pd_tmp = pd.DataFrame(np.array([state_curr,state_bef,state_aft,state_dur,state_diff]).T,columns=['state_current','state_before','state_after','state_dur','state_p_diff'])
            state_bef_aft = pd.concat((state_bef_aft,pd_tmp),ignore_index=True)

        # state_bef_aft['state_match'] = (state_bef_aft['state_after'] == state_bef_aft['state_before'])*1


        # location of PROCESSED state change
        inpt_data_all['K' + str(K) + '_state_processed'] = state_processed
        inpt_data_all['K' + str(K) + '_state_change_processed'] = (inpt_data_all['K' + str(K) + '_state_processed']).diff()
        inpt_data_all.loc[inpt_data_all['K' + str(K) + '_state_change_processed'] != 0,'K' + str(K) + '_state_change_processed'] = 1
        inpt_data_all.loc[inpt_data_all['session_start'] != 0, 'K' + str(K) + '_state_change_processed'] = 1
    # PROCESSED state change index diff
        ## get state duration
    if K == 3:
        state_label = ['LowF-bias', 'HighF-bias', 'Engaged']
    else:
        state_label = ['Disengaged', 'HighF-bias','LowF-bias', 'Engaged']
    # get indices of state change
    state_change_index = list(index_data[inpt_data_all['K' + str(K) + '_state_change_processed'] == 1])
    # get the difference between state changes (hence duration)
    state_change_index_diff = np.diff(state_change_index)
    # add the duration of the last state change
    state_change_index_diff = np.append(state_change_index_diff, len(inpt_data_all) - state_change_index[-1])

    #### check state transitions and state duration
    # get the state label of the according state change
    state_first = inpt_data_all.loc[
        inpt_data_all['K' + str(K) + '_state_change_processed'] == 1, ['K' + str(K) + '_state_processed', 'K' + str(K) + '_diff', 'mouse_id', 'session_identifier']].reset_index(
        drop=True)
    state_first['session'] = state_first['session_identifier'].str.split('s', expand=True)[1]
    # create dataframe for plotting
    state_first['K' + str(K) + '_state_dur'] = state_change_index_diff
    # for i in range(4):
    #     i +=1
    #     plt.hist(state_first.loc[state_first.state_dur==i,'K' + str(K) + '_diff']);plt.show()

    def survey(results, category_names, category_colors):
        """
        Parameters
        ----------
        results : dict
            A mapping from question labels to a list of answers per category.
            It is assumed all lists contain the same number of entries and that
            it matches the length of *category_names*.
        category_names : list of str
            The category labels.
        """
        labels = list(results.keys())
        labels = [str(elements) for elements in labels]
        data = np.array(list(results.values()))


        fig, ax = plt.subplots(1,2,figsize=(14, 12))
        ax[0].invert_yaxis()
        # ax[0].xaxis.set_visible(False)
        ax[0].set_xlim(0, 100)

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            print(i)
            print(colname)
            print(color)
            if i==0:
                starts=0
            elif i==1:
                starts = data[:, i-1]
            else:
                starts = data[:,0] + data[:,1]
            rects = ax[0].barh(labels, widths, left=starts, height=0.5,
                            label=colname, color=color)

            # r, g, b, _ = color
            # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            text_color = 'white'
            ax[0].bar_label(rects, label_type='center', color=text_color)
        ax[0].legend(ncols=len(category_names),
                  loc='lower left', fontsize=14)

        return fig, ax, category_colors
    ## plot dominant state and average transition per session for each mouse
    state_fraction = state_first.groupby(['mouse_id'])['K' + str(K) + '_state_processed'].value_counts(normalize=True).unstack(
        'K' + str(K) + '_state_processed').reset_index()
    state_fraction.iloc[:, 1:] = np.round(state_fraction.iloc[:, 1:] * 100,2)
    state_dict = state_fraction.set_index('mouse_id').T.to_dict('list')

    fig,ax, category_colors=survey(state_dict, state_label, cols)
    ax[0].spines['right'].set_visible(False); ax[0].spines['top'].set_visible(False)
    ax[0].set_xlabel('Percentage of states across mice', fontsize=14);
    # state transition per sess
    state_transition = state_first.groupby(['mouse_id'])['session_identifier'].value_counts().reset_index()
    state_transition_median = state_transition.groupby('mouse_id')['count'].median().reset_index()
    state_transition_median['mouse_id'] = state_transition_median['mouse_id'].astype(str)
    state_transition['mouse_id'] = state_transition['mouse_id'].astype(str)

    ax[1].scatter(state_transition['count'], state_transition['mouse_id'], alpha=0.2, label='individual session')
    ax[1].scatter(state_transition_median['count'], state_transition_median['mouse_id'], color='yellow', edgecolor='k', label='median per mouse')
    ax[1].axvline(x=state_transition_median['count'].median(),linewidth=1,linestyle='--',label='median all')
    ax[1].set_xlabel('Transitions per session', fontsize=14)
    ax[1].legend(loc="lower left", fontsize=14)
    ax[1].spines['left'].set_visible(False); ax[1].spines['top'].set_visible(False)
    ax[1].set_yticks([])
    ax[1].xaxis.set_inverted(True); ax[1].yaxis.set_inverted(True); fig.tight_layout(); fig.show()
    fig.suptitle('State dominance and transitions per mouse', fontsize=14)
    fig.savefig(save_folder/(opto_name + 'K' + str(K) + '_transitions_per_sess_PROCESSED.png'),format='png', bbox_inches="tight")

    ## state duration hist
    fig, ax = plt.subplots(1,len(state_label),figsize=(9,4),sharey=True)
    state_dur_lim = 450
    for i in range(len(state_label)):
        tmp_dur = state_first.loc[(state_first['K' + str(K) + '_state_processed']==i )&
                                  (state_first['K' + str(K) + '_state_dur'] < state_dur_lim), 'K' + str(K) + '_state_dur']
        tmp_dur_median = tmp_dur.median()
        ax[i].hist(tmp_dur,bins=int(state_dur_lim/10), color=category_colors[i],edgecolor='k', alpha=0.5)
        ax[i].axvline(tmp_dur_median, linestyle='--',c='k', label='median = '+ str(int(tmp_dur_median)))
        ax[i].legend(loc="upper right", fontsize='large')
        ax[i].spines['right'].set_visible(False);
        ax[i].spines['top'].set_visible(False)
        ax[i].set_xlim([0,state_dur_lim])
        ax[i].set_xlabel('State dwell times')
    ax[0].set_ylabel('Counts')
    fig.suptitle('Distribution of state dwell times')
    fig.tight_layout(); fig.show()
    fig.savefig(save_folder/(opto_name + 'K' + str(K) + '_state_dwell_times_PROCESSED.png'),format='png', bbox_inches="tight")



    # plot histogram of state probability diff
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(inpt_data_all['K' + str(K) + '_diff'], bins=30, alpha=0.2,edgecolor='black');
    ax.set_ylabel('Counts'); ax.set_xlabel('Difference in posterior probabilities between states', fontsize=14)
    fig.suptitle('Segregation between the most likely state and the next')
    # ax.set_yscale("log");plt.minorticks_off()
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    # https: // stackoverflow.com / questions / 21001088 / how - to - add - different - graphs - as -an - inset - in -another - python - graph
    left, bottom, width, height = [0.2, 0.6, 0.25, 0.25]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.hist(inpt_data_all.loc[inpt_data_all.K3_diff<= 0.2, 'K' + str(K) + '_diff'], bins=30,alpha=0.2, edgecolor='black')

    ax2.spines['right'].set_visible(False);ax2.spines['top'].set_visible(False)
    fig.tight_layout();fig.savefig(save_folder/(opto_name + 'K' + str(K) + '_state_difference.png'),format='png', bbox_inches="tight")

    # plot transitons into and out of
    # K = 3
    n_trials = 3
    # for matrix plot
    fig = plt.figure(figsize=(10, 10))
    columns_label = [str(x) for x in np.arange(K)]
    plotz = K
    index_state_change_all_big = index_data[inpt_data_all['K' + str(K) + '_state_change_processed'] == 1][1:]
    discard_len = []
    discard_len = []
    discard_list = []
    # other states transitioning to 1 state
    for set_K in range(K):
        state_success_lst = []
        state_choice_lst = []
        mean_state_bef_lst = []
        mean_state_aft_lst = []
        # get index of state change of only one state
        index_state_change_all = np.array(
            list(set(index_state_change_all_big).intersection(set(index_data[inpt_data_all['K' + str(K) + '_state_processed'] == set_K]))))
        # get index of the state before the state change
        index_state_end = index_state_change_all - 1
        # get state identity before state change
        state_before = inpt_data_all['K' + str(K) + '_state_processed'].iloc[index_state_end]
        # plt.hist(state_before);
        # plt.show()
        # fig,ax = plt.subplots(1,2,figsize=(10,5))
        # ax[1].hist(state_before,color='k',alpha=0.6);
        # ax[1].set_title('counts of states'); ax[1].set_xlabel('state identity')
        # loop over states before state change
        for k_num in range(K):
            # because theres no state change if state before match current state, set dummy array for plotting
            if k_num == set_K:
                state_success_lst.append(np.zeros(n_trials * 2))
                state_choice_lst.append(np.zeros(n_trials * 2))
                mean_state_bef_lst.append(0)
                mean_state_aft_lst.append(0)
                continue
            # get index of where state before match k_num
            index_state_change_tmp = np.where(state_before == k_num)
            # get real index from the original dataframe
            index_state_change = index_state_change_all[index_state_change_tmp]
            # initialize arrays
            state_start_array_success = np.zeros(n_trials)
            state_end_array_success = np.zeros(n_trials)
            state_start_array_choice = np.zeros(n_trials)
            state_end_array_choice = np.zeros(n_trials)
            mean_state_bef = []
            mean_state_aft = []
            # loop through each state change index to get n_trials before and after
            for i in range(len(index_state_change)):
                # get state change index that matches with the big list
                ind_tmp_big_list = np.where(index_state_change_all_big == index_state_change[i])[0]
                if ind_tmp_big_list == 0:
                    ind_tmp_bef = 1
                    ind_tmp_aft = index_state_change_all_big[ind_tmp_big_list + 1][0]
                elif ind_tmp_big_list == len(index_state_change_all_big) - 1:
                    ind_tmp_bef = index_state_change_all_big[ind_tmp_big_list - 1][0]
                    ind_tmp_aft = len(inpt_data_all) - 1
                else:
                    ind_tmp_bef = index_state_change_all_big[ind_tmp_big_list - 1][0]
                    ind_tmp_aft = index_state_change_all_big[ind_tmp_big_list + 1][0]
                # get length state before and after, if lower than n_trials discard
                len_state_bef = index_state_change[i] - ind_tmp_bef
                len_state_aft = ind_tmp_aft - index_state_change[i]
                if len_state_bef < n_trials or len_state_aft < n_trials:
                    if len_state_bef < n_trials:
                        discard_len.append(len_state_bef)
                        discard_list.append(index_state_change[i] - 1)
                        # inpt_data_all['state_valid_2'].iloc[
                        # index_state_change[i] - len_state_bef:index_state_change[i]] = 0
                    else:
                        discard_len.append(len_state_aft)
                        discard_list.append(index_state_change[i])
                        # inpt_data_all['state_valid_2'].iloc[
                        # index_state_change[i]:index_state_change[i] + len_state_aft] = 0
                    continue
                mean_state_bef.append(np.mean(inpt_data_all['success'].iloc[ind_tmp_bef:index_state_change[i]]))
                mean_state_aft.append(np.mean(inpt_data_all['success'].iloc[index_state_change[i]:ind_tmp_aft]))

                # state end mean and length
                # start and end of the state change with success
                state_start_array_success = np.vstack(
                    [state_start_array_success,
                     inpt_data_all['success'].iloc[index_state_change[i]:index_state_change[i] + n_trials]])
                state_end_array_success = np.vstack(
                    [state_end_array_success,
                     inpt_data_all['success'].iloc[index_state_change[i] - n_trials:index_state_change[i]]])
                # start and end of the state change with choice
                state_start_array_choice = np.vstack(
                    [state_start_array_choice,
                     inpt_data_all['choice'].iloc[index_state_change[i]:index_state_change[i] + n_trials]])
                state_end_array_choice = np.vstack(
                    [state_end_array_choice,
                     inpt_data_all['choice'].iloc[index_state_change[i] - n_trials:index_state_change[i]]])
            # list of states
            state_success_lst.append(np.hstack([state_end_array_success[1:, :], state_start_array_success[1:, :]]))
            state_choice_lst.append(np.hstack([state_end_array_choice[1:, :], state_start_array_choice[1:, :]]))
            mean_state_bef_lst.append(np.mean(np.array(mean_state_bef)))
            mean_state_aft_lst.append(np.mean(np.array(mean_state_aft)))
        for i in range(plotz):
            if i == set_K:
                # initialize ax
                ax = plt.subplot2grid((plotz, plotz), (i, set_K), fig=fig)
                # set color for this subgrid
                ax.set_facecolor("#580F41")
                ax.patch.set_alpha(0.9)
                ax.yaxis.set_ticklabels([])
                ax.xaxis.set_ticklabels([])
                if i == plotz - 1:
                    ax.set_xlabel('to ' + columns_label[set_K])
                if set_K == 0:
                    ax.set_ylabel('from ' + columns_label[set_K], rotation=0, ha='right')
                # Show all ticks and label them with the respective list entries

            else:
                print(i, set_K)
                ax = plt.subplot2grid((plotz, plotz), (i, set_K), fig=fig)
                ax.plot(np.mean(state_success_lst[i], axis=0), color=cols[i], lw=1.2)
                ax.plot(np.mean(state_choice_lst[i], axis=0), color=cols[i], lw=1, linestyle='--')
                ax.axhline(y=mean_state_bef_lst[i], xmin=0, xmax=0.4, linestyle='--', lw=.6, color=cols[i])
                ax.axhline(y=mean_state_aft_lst[i], xmin=0.6, xmax=1, linestyle='--', lw=.6, color=cols[set_K])
                ax.axvline(x=n_trials, linestyle='--', lw=.5, color='k');
                ax.axhline(y=0.5, linestyle='--', lw=.5, color='k')
                ax.set_xticks(np.arange(n_trials*2))
                ax.xaxis.set_ticklabels(['t-3','t-2','t-1','t','t+1','t+2'])
                if i == plotz - 1:
                    ax.set_xlabel('to ' + columns_label[set_K])
                else:
                    ax.xaxis.set_ticklabels([])
                if set_K == 0:
                    ax.set_ylabel('from ' + columns_label[i], rotation=0, ha='right')
                else:
                    ax.yaxis.set_ticklabels([])
            ax.set_ylim([-0.01, 1.01])
            ax.set_xlim([-0.1, n_trials * 2 - .9])
    fig.savefig(save_folder/(opto_name + 'K' + str(K) + '_transition_into_out_matrix.png'), format='png', bbox_inches="tight")
    plt.show()

    # #### plot psychometrics based on states
    # state_label = ['LowF-bias','HighF-bias','Engaged']
    # # conditioned on pfail
    # data_stack = inpt_data_all.groupby(['mouse_id','K3_state_processed','binned_freq','pfail'])['choice'].value_counts(
    #             normalize=True).unstack('choice').reset_index()
    # g = sns.FacetGrid(data_stack, row='K3_state_processed',hue='pfail', height=3.5, aspect=.95)
    # g.map_dataframe(sns.lineplot, x='binned_freq', y=0);axes = g.axes.flatten()
    # for i, ax in enumerate(axes):
    #     ax.set_ylim([0, 1]);
    #     ax.set_xlim([-.75, .75])
    #     ax.set_ylabel('Proportion High')
    #     ax.axhline(0.5, ls='--', c='black', linewidth=0.2)
    #     ax.axvline(0, ls='--', c='black', linewidth=0.2)
    #     ax.set_title(state_label[i])
    #     if i == 2:
    #         ax.set_xlabel('Distance to threshold (Octave)')
    #         ax.legend(loc='lower right',title='Pfail')
    # plt.show()
    # g.savefig(save_folder / ('psychometrics_state_pfail_conditioned.png'), format='png', bbox_inches="tight" )
    #
    # # conditioned on pfail and pchoice
    # data_stack = inpt_data_all.groupby(['mouse_id', 'K3_state_processed', 'binned_freq', 'pfail','pchoice'])[
    #     'choice'].value_counts(
    #     normalize=True).unstack('choice').reset_index()
    # state_label = ['LowF-bias','LowF-bias', 'HighF-bias', 'HighF-bias', 'Engaged', 'Engaged']
    # g = sns.FacetGrid(data_stack,
    #                   row='K3_state_processed', col='pfail', hue='pchoice', palette=['c','y'], height=3.5, aspect=.95)
    # g.map_dataframe(sns.lineplot, x='binned_freq', y=0);
    # axes = g.axes.flatten()
    # for i, ax in enumerate(axes):
    #     ax.set_ylim([0, 1]);
    #     ax.set_xlim([-.75, .75])
    #     ax.set_ylabel('Proportion High')
    #     ax.set_title('')
    #     ax.axhline(0.5, ls='--', c='black', linewidth=0.2)
    #     ax.axvline(0, ls='--', c='black', linewidth=0.2)
    #     if i >= len(axes)-2:
    #         ax.set_xlabel('Distance to threshold (Octave)')
    # axes[1].set_title('Pfail')
    # axes[0].set_title('Psuccess')
    # plt.show()
    # g.savefig(save_folder / ('psychometrics_state_pfail_pchoice_conditioned.png'), format='png', bbox_inches="tight")
    #
    #
    # ####
    # coef_df = pd.read_csv('/home/anh/Documents/phd/outcome_manip/data/glmer_state_transition_coef.csv')
    # model_name_lst = coef_df.model_name.unique()
    # pred_col_lst = coef_df.columns[1:6]
    # pred_col_label = ['stim','pchoice','pfail','rb','ra']
    # for i in range(len(model_name_lst)):
    #     fig,ax = plt.subplots(figsize=(4,4))
    #     ax.plot(np.arange(len(pred_col_lst)), coef_df.loc[coef_df.model_name==model_name_lst[i],pred_col_lst].values.squeeze(),'-ko', linewidth=0.5)
    #     ax.axhline(y=0,linewidth=0.2,linestyle='--')
    #     ax.spines['right'].set_visible(False);
    #     ax.spines['top'].set_visible(False)
    #     ax.set_xticks(np.arange(len(pred_col_lst)))
    #     ax.set_xticklabels(pred_col_label,rotation=45)
    #     ax.set_title(label=model_name_lst[i],loc='left')
    #     fig.savefig(save_folder / ('K3_transition_coef_'+model_name_lst[i]+'.png'), format='png', bbox_inches="tight")

    # # plot diff state conditioned on pre/during/post transition point
    # state_change_index = list(index_data[inpt_data_all.K3_state_change==1][1:])
    # num_col = 5
    # num_row = 3
    # fig,ax=plt.subplots(num_row,num_col,figsize=(24,18),sharey=True)
    # for sc_id in range(len(state_change_index)):
    #     for row_id in range(num_row):
    #         shift_diff_index = row_id - 1
    #         ax_plot = 1
    #         transition_diff = inpt_data_all.loc[state_change_index[sc_id]+shift_diff_index,'K3_diff']
    #         if transition_diff > 0.8:
    #             ax_plot = 4
    #         elif transition_diff < 0.2:
    #             ax_plot = 0
    #         elif (transition_diff <= 0.55) & (transition_diff >= 0.45):
    #             ax_plot= 2
    #         elif (transition_diff > 0.55) & (transition_diff <= 0.8):
    #             ax_plot = 3
    #
    #         ax[row_id,ax_plot].plot(np.arange(7),inpt_data_all.loc[state_change_index[sc_id]-3:state_change_index[sc_id]+3,'K3_diff'],linestyle='--',color='grey',linewidth=0.2)
    # for col_id in range(num_col):
    #     for row_id in range(num_row):
    #         ax[row_id,col_id].axvline(3,linestyle='--',color='r')
    #         ax[row_id,col_id].set_xticks(np.arange(7))
    #         ax[row_id,col_id].set_xticklabels(labels=['t-3', 't-2', 't-1', 't', 't+1', 't+2', 't+3']);
    # ax[0,0].set_title('transitions conditioned on t-1 difference')
    # ax[1, 0].set_title('transitions conditioned on t difference')
    # ax[2, 0].set_title('transitions conditioned on t+1 difference')
    # fig.tight_layout(); fig.show()
    # # fig.savefig(save_folder / ('K3_transition_points_various_conditioned.png'),format='png', bbox_inches="tight")
    # del fig
    #
    # # plot histogram of state probability diff AT STATE TRANSITION
    # fig, ax = plt.subplots(figsize=(6, 4))
    # tmp_data = inpt_data_all.loc[inpt_data_all.K3_state_change==1].copy()
    # ax.hist(tmp_data['K3_diff'], bins=30, alpha=0.2, color='k',edgecolor='black');
    # ax.set_ylabel('Counts');
    # ax.set_xlabel('Difference in posterior probabilities between states')
    # ax.set_title('Segregation between the most likely state and the next\n at transition point')
    # ax.spines['right'].set_visible(False);
    # ax.spines['top'].set_visible(False)
    # # https: // stackoverflow.com / questions / 21001088 / how - to - add - different - graphs - as -an - inset - in -another - python - graph
    # left, bottom, width, height = [0.2, 0.6, 0.25, 0.25]
    # ax2 = fig.add_axes([left, bottom, width, height])
    # ax2.set_title('Short states',fontsize=8,loc='left')
    # ax2.hist(tmp_data.loc[tmp_data.state_dur <= 3, 'K3_diff'], color='k', bins=30, alpha=0.2, edgecolor='black')
    # ax2.spines['right'].set_visible(False);
    # ax2.spines['top'].set_visible(False)
    # fig.tight_layout();fig.show()
    # fig.savefig(save_folder / ('K3_state_difference_at_transition.png'), format='png', bbox_inches="tight")
    # del tmp_data
    #
    # # plot diff state with counts of states at transition point
    # num_row = 1
    # p_diff_seg = [0, 0.2, 0.45, 0.55, 0.8, 1]
    # state_dur_lst = [1,2,3,4]
    # for state_dur_id in range(len(state_dur_lst)):
    #     if state_dur_id == 3:
    #         state_change_index = list(index_data[(inpt_data_all.K3_state_change == 1) &
    #                                              (inpt_data_all.state_dur >= state_dur_lst[state_dur_id])])[1:]
    #         linewidth_num = 0.1
    #         title_fig = 'Transitions of states with duration of more than ' + str(state_dur_lst[state_dur_id]) + \
    #                     ' trial(s) (n = ' + str(len(state_change_index)) + ')'
    #     else:
    #         state_change_index = list(index_data[(inpt_data_all.K3_state_change == 1) &
    #                                              (inpt_data_all.state_dur == state_dur_lst[state_dur_id])])
    #         linewidth_num = 0.5
    #         title_fig = 'Transitions of states with duration of ' + str(
    #             state_dur_lst[state_dur_id]) + ' trial(s) (n = ' + str(len(state_change_index)) + ')'
    #     fig, ax = plt.subplots(num_row, num_col, figsize=(24, 6 * num_row), sharey='row')
    #     state_lst = [[0], [0], [0], [0], [0]]
    #     for sc_id in range(len(state_change_index)):
    #         transition_diff = inpt_data_all.loc[state_change_index[sc_id], 'K3_diff']
    #         transition_state = inpt_data_all.loc[state_change_index[sc_id], 'K3_state']
    #         if transition_diff >= 0.8:
    #             ax_plot = 4
    #         elif transition_diff < 0.2:
    #             ax_plot = 0
    #         elif (transition_diff < 0.55) & (transition_diff >= 0.45):
    #             ax_plot = 2
    #         elif (transition_diff >= 0.55) & (transition_diff < 0.8):
    #             ax_plot = 3
    #         else:
    #             ax_plot = 1
    #
    #         state_lst[ax_plot].extend([transition_state])
    #         ax[ax_plot].plot(np.arange(7),
    #                          inpt_data_all.loc[state_change_index[sc_id] - 3:state_change_index[sc_id] + 3,
    #                          'K3_diff'], linestyle='--', color='grey', linewidth=linewidth_num)
    #     ax[0].set_ylabel('Posterior probabilities difference')
    #     for col_id in range(num_col):
    #         ax[col_id].axvline(3, linestyle='--', color='r')
    #         ax[col_id].set_xticks(np.arange(7))
    #         ax[col_id].set_xticklabels(labels=['t-3', 't-2', 't-1', 't', 't+1', 't+2', 't+3']);
    #
    #     fig.suptitle(title_fig); fig.tight_layout(); fig.show()
    #     fig.savefig(save_folder / ('K3_transition_points_state_diff_state_dur_'+str(state_dur_lst[state_dur_id])+'.png'), format='png', bbox_inches="tight")
    #
    #     fig, ax = plt.subplots(1, num_col, figsize=(24, 5), sharey='row')
    #     for hist_id in range(num_col):
    #         # this is to get rid of the initialized 0s
    #         to_parse = state_lst[hist_id][1:]
    #         count_state = Counter(to_parse)
    #         count_state = dict(sorted(count_state.items()))
    #         # get fraction state match
    #         state_tmp = state_bef_aft.loc[(state_bef_aft.state_dur == state_dur_lst[state_dur_id]) &
    #                                       (state_bef_aft.state_p_diff >= p_diff_seg[hist_id]) &
    #                                       (state_bef_aft.state_p_diff < p_diff_seg[hist_id+1])].copy()
    #
    #         state_tmp_stack = state_tmp.groupby(['state_current'])['state_match'].value_counts(normalize=True).unstack(
    #             'state_match').reset_index()
    #
    #         # plot using dict keys and values for bars
    #         ax[hist_id].bar(np.array(list(count_state.keys())), count_state.values(), alpha=0.2, color='black',
    #                             width=0.5, label = 'state counts')
    #         # this is to prevent no data acquired
    #         if len(state_tmp) != 0 and state_dur_id !=3:
    #             fraction_state_match = np.array(list(count_state.values()))*np.array(state_tmp_stack[1].values)
    #         else:
    #             fraction_state_match = 0
    #         # dont plot this if state_dur > 4
    #         if state_dur_id != 3:
    #             ax[hist_id].bar(np.array(list(count_state.keys())), fraction_state_match,
    #                             width=0.5, edgecolor='k', align='center', alpha=0.2, hatch='//', label = 'fraction match')
    #
    #         ax[hist_id].set_xticks(np.arange(3));
    #         ax[hist_id].set_xticklabels(labels=['LowF-bias', 'HighF-bias', 'Engaged']);
    #         if hist_id == 0:
    #             ax[hist_id].set_ylabel('Counts')
    #     ax[hist_id].legend(loc='upper right')
    #     fig.tight_layout();
    #     fig.show()
    #     fig.savefig(save_folder / ('K3_transition_points_state_counts_'+str(state_dur_lst[state_dur_id])+'.png'), format='png', bbox_inches="tight")
    #
