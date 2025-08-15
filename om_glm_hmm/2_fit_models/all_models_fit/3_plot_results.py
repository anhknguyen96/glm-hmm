import json
from pathlib import Path
import os
import numpy as np
import sys
import pandas as pd
sys.path.append('.')
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns
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
pfail = 0
if pfail == 1:
    root_folder_name = 'om_choice'
    # list of columns/predictors name as ordered by pansy
    labels_for_plot = ['pfail', 'stim', 'stim_pfail', 'pchoice', 'bias']
else:
    root_folder_name = 'om_choice_nopfail_opto'
    # list of columns/predictors name as ordered by pansy
    labels_for_plot = ['stim', 'pchoice', 'bias']
    # separate because the upper one is involved in column names
    labels_for_plot_weights = ['bias', 'stim', 'pchoice']
state_label = ['LowF-bias','HighF-bias','Engaged','Non']
save_folder = Path(root_folder_dir) / root_folder_name / 'plots'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
root_data_dir = Path(root_folder_dir) / root_folder_name / 'data'
root_result_dir = Path(root_folder_dir) / root_folder_name / 'result'
root_result_1_dir = Path(root_folder_dir) / root_folder_name / 'result_transitionalpha_1-2'
root_result_3_dir = Path(root_folder_dir) / root_folder_name / 'result_transitionalpha_3-5'

data_dir = root_data_dir / (root_folder_name +'_data_for_cluster')
data_individual = data_dir / 'data_by_animal'
results_dir = root_result_dir / (root_folder_name +'_global_fit')
results_dir_individual = root_result_dir/ (root_folder_name + '_individual_fit')
results_dir_1_individual = root_result_1_dir / (root_folder_name + '_individual_fit')
results_dir_3_individual = root_result_3_dir / (root_folder_name + '_individual_fit')
animal_list = load_animal_list(data_individual / 'animal_list.npz')

# first index for all animals
n_session_lst = [1,50]              # higher n sessions (>=60) seems to yield poorer fit??
n_trials_lst = [5000,250]
# cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306",
#         '#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
#         '#999999', '#e41a1c', '#dede00'
#     ]

cols = [ "#15b01a", "#f97306", "#7e1e9c", "#3498db"
    ]
# trouble_animals =['23.0','24.0','26.0']
trouble_animals =[]
animal_list = list(set(animal_list)-set(trouble_animals))

# flag for running all_animals analysis
all_animals = 0
# flag for running individual animals analysis
individual_animals = 0
# flag for running k-state glm fit check
glm_fit_check = 0
sim_data = 0
# flag for predictive accuracy plot
pred_acc_plot = 0
pred_acc_plot_multialpha = 0
# flag for one animal
one_animal = 0

exploratory_plot = 1

####################################################################################
##################### K-STATE GLM PRED ACC & NLL ###################################
if pred_acc_plot_multialpha:
    K = 5
    K_plot = 5
    D, M, C = 1, 3, 2

    plt_xticks_location = np.arange(3)
    plt_xticks_label =  ['standard', r"$\Delta$ lower", r"$\Delta$ higher"]
    idx_cv_arr = [1,2]


    # =========== NLL ====================================
    # plt.subplot(3, 3, 1)
    fig, ax = plt.subplots(figsize=(5, 3))
    across_animals = np.zeros((1,4))
    for animal in animal_list:
        results_dir_individual_animal = results_dir_individual / animal
        cv_arr = load_cv_arr(results_dir_individual_animal / "cvbt_folds_model.npz")
        cv_arr_1 = load_cv_arr(results_dir_1_individual / animal / "cvbt_folds_model.npz")
        cv_arr_3 = load_cv_arr(results_dir_3_individual / animal / "cvbt_folds_model.npz")
        mean_cvbt = np.mean(cv_arr, axis=1)
        mean_cvbt_1 = np.mean(cv_arr_1, axis=1)
        mean_cvbt_3 = np.mean(cv_arr_3, axis=1)

        tmp_mean = np.stack((mean_cvbt,mean_cvbt_1,mean_cvbt_3))
        tmp_mean = (tmp_mean - tmp_mean[0,:]).T
        tmp_mean[:, 0] = int(float(animal)) * np.ones(K)
        tmp_mean = np.hstack((tmp_mean,np.zeros((K,1))))
        tmp_mean[:,-1] = np.arange(K)+1
        # ax.plot(plt_xticks_location, tmp_mean[:,1],color='k',lw=1.5)
        across_animals = np.vstack((across_animals,tmp_mean))
    # plt.xticks(plt_xticks_location, plt_xticks_label,
    #            fontsize=10)
    across_animals = across_animals[1:,:]
    delta_pd = pd.DataFrame(across_animals, columns=['animal_id','lower','higher','K-state'])
    delta_pd = pd.melt(delta_pd, id_vars=['animal_id','K-state'], value_vars=['lower', 'higher'], var_name='delta_alpha', value_name='delta_LL')
    # delta_pd['delta_LL'] = delta_pd['delta_LL'].round(2)
    sns.swarmplot(data=delta_pd, x='K-state', y='delta_LL', hue='delta_alpha', dodge=True,ax=ax); plt.tight_layout()
    fig.savefig(save_folder / ('fig4_a' + 'delta_K_' + str(K_plot) + '_all.png'), format='png', bbox_inches="tight")
    plt.show()
    # plt.show()

####################################################################################
##################### K-STATE GLM PRED ACC & NLL ###################################
if pred_acc_plot:
    K = 5
    K_plot = 5
    D, M, C = 1, 3, 2

    plt_xticks_location = np.arange(K_plot)
    plt_xticks_label =  [str(x) for x in (plt_xticks_location + 1)]
    idx_cv_arr = [1,2]

    global_weights = get_global_weights(results_dir, K_plot)

    # =========== NLL ====================================
    # plt.subplot(3, 3, 1)
    fig, ax = plt.subplots(figsize=(3, 3))
    across_animals = []
    for animal in animal_list:
        results_dir_individual_animal = results_dir_individual / animal
        cv_arr = load_cv_arr(results_dir_individual_animal / "cvbt_folds_model.npz")
        # idx_range = len(cv_arr) - (K-K_plot)
        # idx_tmp = np.arange(idx_range)
        # idx = np.delete(idx_tmp, idx_cv_arr)
        # cv_arr_for_plotting = cv_arr[idx, :]
        mean_cvbt = np.mean(cv_arr, axis=1)
        across_animals.append(mean_cvbt - mean_cvbt[0])
        plt.plot(plt_xticks_location,
                     mean_cvbt - mean_cvbt[0],
                     '-o',
                     color='#999999',
                     zorder=0,
                     alpha=0.6,
                     lw=1.5,
                     markersize=4)
    across_animals = np.array(across_animals)
    mean_cvbt = np.mean(np.array(across_animals), axis=0)
    plt.plot(plt_xticks_location,
             mean_cvbt - mean_cvbt[0],
             '-o',
             color='k',
             zorder=1,
             alpha=1,
             lw=1.5,
             markersize=4,
             label='mean')
    plt.xticks(plt_xticks_location, plt_xticks_label,
               fontsize=10)
    plt.ylabel("$\Delta$ test LL (bits/trial)", fontsize=10, labelpad=0)
    plt.xlabel("# states", fontsize=10, labelpad=0)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylim((-0.01, 0.24))
    # plt.yticks(color = cols[0])
    leg = plt.legend(fontsize=10,
                     labelspacing=0.05,
                     handlelength=1.4,
                     borderaxespad=0.05,
                     borderpad=0.05,
                     framealpha=0,
                     bbox_to_anchor=(1.2, 0.90),
                     loc='lower right',
                     markerscale=0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)
    fig.savefig(save_folder / ('fig4_a'+'K_'+str(K_plot)+'_all.png'),format='png', bbox_inches="tight")
    # =========== PRED ACC =========================
    # plt.subplot(3, 3, 2)
    fig, ax = plt.subplots(figsize=(3, 3))
    mean_across_animals = []
    num_trials_all_animals = 0
    for z, animal in enumerate(animal_list):
        results_dir_individual_animal = results_dir_individual / animal

        correct_mat, num_trials = load_correct_incorrect_mat(
            results_dir_individual_animal / "correct_incorrect_mat.npz")
        num_trials_all_animals += np.sum(num_trials)
        if z == 0:
            trials_correctly_predicted_all_folds = np.sum(correct_mat, axis=1)
        else:
            trials_correctly_predicted_all_folds = \
                trials_correctly_predicted_all_folds + np.sum(
                correct_mat, axis=1)

        pred_acc_arr = load_cv_arr(results_dir_individual_animal / "predictive_accuracy_mat.npz")
        # this is because predictive accuracy is not calculated when lapse model is not calculated
        pred_acc_arr_for_plotting = pred_acc_arr[plt_xticks_location,:]
        mean_acc = np.mean(pred_acc_arr_for_plotting, axis=1)
        plt.plot(plt_xticks_location,
                     mean_acc - mean_acc[0],
                     '-o',
                     color='#EEDFCC',
                     zorder=0,
                     alpha=0.6,
                     lw=1.5,
                     markersize=4)
        mean_across_animals.append(mean_acc - mean_acc[0])
    ymin = -0.01
    ymax = 0.15
    plt.xticks(plt_xticks_location, plt_xticks_label,
               fontsize=10)
    plt.ylim((ymin, ymax))
    trial_nums = (trials_correctly_predicted_all_folds -
                  trials_correctly_predicted_all_folds[0])
    plt.plot(plt_xticks_location,
             np.mean(mean_across_animals, axis=0),
             '-o',
             color='#8B8378',
             zorder=0,
             alpha=1,
             markersize=4,
             lw=1.5,
             label='mean')
    plt.ylabel("$\Delta$ pred acc. (%)", fontsize=10, labelpad=0)
    plt.xlabel("# states", fontsize=10, labelpad=0)
    plt.yticks([0, 0.05, 0.1], ["0", "5%", '10%'])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(save_folder / ('fig4_b'+'K_'+str(K_plot)+'_all.png'),format='png', bbox_inches="tight")

##################### K-STATE GLM FIT CHECK ########################################
##################### SIMULATE VEC #################################################
if glm_fit_check:
    # plot psychometrics of simulated data
    if sim_data:
        n_trials = n_trials_lst[0]
        # simulate stim vec
        stim_vec_sim = simulate_stim(n_trials+1)
        # z score stim vec
        z_stim_sim = (stim_vec_sim - np.mean(stim_vec_sim)) / np.std(stim_vec_sim)
        # define col names for simulated dataframe
        col_names = labels_for_plot + ['choice','outcome','stim_org','state','animal']
        # psychometrics stim list - since min/max freq_trans is -1.5/1.5
        bin_lst = np.arange(-1.55, 1.6, 0.1)
        bin_name = np.round(np.arange(-1.5, 1.6, .1), 2)
        # iterate over all k-states model
        for K in range(2,K_max+1):
            # create fig
            fig, ax= plt.subplots(3,K,figsize=(K+4,8),sharey="row")
            # initialize simulated array
            inpt_sim = np.zeros(len(col_names)).reshape(1, -1)
            # get global weights - the function also permutate the states
            global_weight_vectors = get_global_weights(results_dir, K)
            # get global transition matrix
            global_transition_matrix = get_global_trans_mat(results_dir, K)
            # get global state dwell times
            global_dwell_times = 1 / (np.ones(K) - global_transition_matrix.diagonal())
            state_dwell_times = np.zeros((len(animal_list)+1, K))
            # iterate over k-state to simulate glm data
            for k_ind in range(K):
                # simulate input array and choice
                inpt_sim_tmp = simulate_from_weights_pfailpchoice_model(np.squeeze(global_weight_vectors[k_ind,:,:]),n_trials,z_stim_sim,stim_vec_sim,pfail)
                # add original simulated stim vec
                inpt_sim_tmp = np.append(inpt_sim_tmp, np.array(stim_vec_sim[:-1]).reshape(-1,1), axis=1)
                # add state info
                inpt_sim_tmp = np.append(inpt_sim_tmp, k_ind * np.ones(inpt_sim_tmp.shape[0]).reshape(-1, 1), axis=1)
                # add animal info
                inpt_sim_tmp = np.append(inpt_sim_tmp, np.zeros(inpt_sim_tmp.shape[0]).reshape(-1, 1), axis=1)
                # stack simulated input for each k
                inpt_sim = np.vstack((inpt_sim, inpt_sim_tmp))

            # now do it for separate animals
            for k_ind in range(K):
                for z,animal in enumerate(animal_list):
                    results_dir_individual_animal = results_dir_individual / animal

                    cv_file = results_dir_individual_animal / "cvbt_folds_model.npz"
                    cvbt_folds_model = load_cv_arr(cv_file)

                    with open(results_dir_individual_animal / "best_init_cvbt_dict.json", 'r') as f:
                        best_init_cvbt_dict = json.load(f)

                    # Get the file name corresponding to the best initialization for
                    # given K value
                    raw_file = get_file_name_for_best_model_fold(
                        cvbt_folds_model, K, results_dir_individual_animal, best_init_cvbt_dict)
                    hmm_params, lls = load_glmhmm_data(raw_file)
                    weight_vectors = -hmm_params[2]
                    transition_matrix = np.exp(hmm_params[1][0])
                    # state dwell times for individual animals
                    state_dwell_times[z, :] = 1 / (np.ones(K) -
                                                   transition_matrix.diagonal())

                    # # simulate input array and choice
                    inpt_sim_tmp = simulate_from_weights_pfailpchoice_model(np.squeeze(weight_vectors[k_ind, :, :]), n_trials,
                                                                            z_stim_sim,stim_vec_sim,pfail)
                    # add original simulated stim vec
                    inpt_sim_tmp = np.append(inpt_sim_tmp, np.array(stim_vec_sim[:-1]).reshape(-1, 1), axis=1)
                    # add state info
                    inpt_sim_tmp = np.append(inpt_sim_tmp, k_ind * np.ones(inpt_sim_tmp.shape[0]).reshape(-1, 1), axis=1)
                    # add animal info
                    inpt_sim_tmp = np.append(inpt_sim_tmp, int(float(animal))*np.ones(inpt_sim_tmp.shape[0]).reshape(-1, 1), axis=1)
                    # stack simulated input for each k
                    inpt_sim = np.vstack((inpt_sim, inpt_sim_tmp))

                    # plot individual weights here
                    ax[0, k_ind].plot(labels_for_plot, np.squeeze(weight_vectors[k_ind, :, :]), '--', color=cols[k_ind])

                ######################## K-STATE WEIGHTS ################################################
                ax[0, k_ind].plot(labels_for_plot, np.squeeze(global_weight_vectors[k_ind, :, :]), label='global',
                                  color='black', linewidth=1.5)
                ax[0, k_ind].set_xticklabels(labels_for_plot, fontsize=12, rotation=45)
                ax[0, k_ind].axhline(0, linewidth=0.5, linestyle='--')
                ax[0, k_ind].set_title('state ' + str(k_ind))
                ######################## DWELL TIMES ####################################################
                logbins = np.logspace(np.log10(1),
                                      np.log10(max(state_dwell_times[:, k_ind])), 15)
                # plot dwell times for each state
                ax[2,k_ind].hist(state_dwell_times[:, k_ind],
                         bins=logbins,
                         color=cols[k_ind],
                         histtype='bar',
                         rwidth=0.8)
                ax[2,k_ind].axvline(np.median(state_dwell_times[:, k_ind]),
                            linestyle='--',
                            color='k',
                            lw=1,
                            label='median')
                # if k_ind == 0:
                #     ax[2,k_ind].set_ylabel("# animals", fontsize=12)
                # ax[2,k_ind].set_xlabel("expected dwell time \n (# trials)",
                #            fontsize=12)

            # create dataframe
            inpt_sim_df = pd.DataFrame(data=inpt_sim[1:,:],columns=col_names)

            ##################### PSYCHOMETRIC CURVES ##########################################
            # get binned freqs for psychometrics
            inpt_sim_df["binned_freq"] = pd.cut(inpt_sim_df.stim_org, bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)
            sim_stack = inpt_sim_df.groupby(['binned_freq','state','animal'])['choice'].value_counts(normalize=True).unstack('choice').reset_index()
            sim_stack[-1] = sim_stack[-1].fillna(0)
            # sns.lineplot(sim_stack.binned_freq,sim_stack[-1],hue=sim_stack.state);plt.show()
            ##################### PLOT PSYCHOMETRICS FOR EACH K-STATE ######################################
            for k_ind in range(K):
                for animal in animal_list:
                    ax[1,k_ind].plot(sim_stack.binned_freq.unique(),sim_stack[-1].loc[(sim_stack.state==k_ind)&(sim_stack.animal==int(float(animal)))],'--',color=cols[k_ind])
                ax[1, k_ind].plot(sim_stack.binned_freq.unique(), sim_stack[-1].loc[(sim_stack.state == k_ind)&(sim_stack.animal==0)],color='black')
                ax[1, k_ind].set_xticklabels(labels= [str(x) for x in sim_stack.binned_freq.unique()] ,fontsize=12, rotation=45)
                ax[1, k_ind].axhline(0.5, linewidth=0.5, linestyle='--')
                ax[1, k_ind].axvline(6, linewidth=0.5, linestyle='--')
            plt.tight_layout()
            plt.show()
            fig.savefig(save_folder / ('all_K'+ str(K) + '_glmhmm_modelcheck.png'),format='png',bbox_inches = "tight")
    # plot psychometrics of real data
    else:

        # psychometrics stim list - since min/max freq_trans is -1.5/1.5
        bin_lst = np.arange(-1.55, 1.6, 0.1)
        bin_name = np.round(np.arange(-1.5, 1.6, .1), 2)
        # load dictionary for best cv model run
        with open(results_dir / "best_init_cvbt_dict.json", 'r') as f:
            global_best_init_cvbt_dict = json.load(f)
        # load cv array
        global_cv_file = results_dir / 'cvbt_folds_model.npz'
        global_cvbt_folds_model = load_cv_arr(global_cv_file)

        # load animal data for simulation
        inpt, y, session = load_data(data_dir / 'all_animals_concat.npz')
        inpt_unnorm, _, _ = load_data(data_dir / 'all_animals_concat_unnormalized.npz')
        # create dataframe all animals for plotting
        inpt_unnorm = np.append(inpt_unnorm, np.ones(inpt_unnorm.shape[0]).reshape(-1, 1), axis=1)
        global_inpt_data_all = pd.DataFrame(data=inpt_unnorm, columns=labels_for_plot)
        global_inpt_data_all['choice'] = y
        # prepare data
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                       inpt.shape[0])
        y[np.where(y == -1), :] = 1
        global_inputs, global_datas, global_train_masks = partition_data_by_session(
            np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

        # iterate over all k-states model
        for K in range(2, K_max + 1):
            # create fig
            fig, ax = plt.subplots(2, K, figsize=(K + 4, 6), sharey="row")
            # fig, ax = plt.subplots(3, K, figsize=(K + 4, 8), sharey="row")
            inpt_data_all = pd.DataFrame()
            # get global raw file for each state
            raw_file = get_file_name_for_best_model_fold(
                global_cvbt_folds_model, K, results_dir, global_best_init_cvbt_dict)
            hmm_params, lls = load_glmhmm_data(raw_file)
            # get global weights - the function also permutate the states
            global_weight_vectors = get_global_weights(results_dir, K)
            # get global transition matrix
            global_transition_matrix = get_global_trans_mat(results_dir, K)
            # get global state dwell times
            global_dwell_times = 1 / (np.ones(K) - global_transition_matrix.diagonal())
            state_dwell_times = np.zeros((len(animal_list) + 1, K))

            # get posterior probs for state inference
            permutation_to_align_individual_animals = calculate_state_permutation(hmm_params)
            posterior_probs = get_marginal_posterior(global_inputs, global_datas, global_train_masks,
                                                     hmm_params, K, permutation_to_align_individual_animals)
            states_max_posterior = np.argmax(posterior_probs, axis=1)
            global_inpt_data_all['state'] = states_max_posterior
            global_inpt_data_all['animal'] = np.zeros(len(global_inpt_data_all))

            # add global data info to indiv data frame
            inpt_data_all = pd.concat([inpt_data_all, global_inpt_data_all],ignore_index=True)
            # now do it for separate animals
            for z, animal in enumerate(animal_list):
                print(animal)
                results_dir_individual_animal = results_dir_individual / animal
                cv_file = results_dir_individual_animal / "cvbt_folds_model.npz"
                cvbt_folds_model = load_cv_arr(cv_file)

                with open(results_dir_individual_animal / "best_init_cvbt_dict.json", 'r') as f:
                    best_init_cvbt_dict = json.load(f)

                # Get the file name corresponding to the best initialization for given K value
                raw_file = get_file_name_for_best_model_fold(
                    cvbt_folds_model, K, results_dir_individual_animal, best_init_cvbt_dict)
                hmm_params, lls = load_glmhmm_data(raw_file)
                weight_vectors = -hmm_params[2]
                transition_matrix = np.exp(hmm_params[1][0])
                # state dwell times for individual animals
                state_dwell_times[z, :] = 1 / (np.ones(K) -
                                               transition_matrix.diagonal())
                # Also get data for animal:
                inpt, y, session = load_data(data_individual / (animal + '_processed.npz'))
                inpt_unnorm, _, _ = load_data(data_individual / (animal + '_unnormalized.npz'))
                all_sessions = np.unique(session)
                # create dataframe single animals for plotting
                inpt_unnorm = np.append(inpt_unnorm, np.ones(inpt_unnorm.shape[0]).reshape(-1, 1), axis=1)
                inpt_data = pd.DataFrame(data=inpt_unnorm, columns=labels_for_plot)
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
                inpt_data['animal'] = np.ones(len(inpt_data)) * int(float(animal))
                # stack data from individual animals
                inpt_data_all = pd.concat([inpt_data_all, inpt_data], ignore_index=True)

                # plot individual weights here
                for k_ind in range(K):
                    new_ordered_weights_tmp = np.squeeze(weight_vectors[k_ind, :, :])
                    bias_weight = new_ordered_weights_tmp[-1]
                    new_ordered_weights_tmp = np.insert(new_ordered_weights_tmp, 0, bias_weight)
                    ax[0, k_ind].plot(labels_for_plot_weights, new_ordered_weights_tmp[:-1], '--', color=cols[k_ind])
                del inpt, inpt_data, inpt_unnorm, y

            ##################### PSYCHOMETRIC CURVES ##########################################
            # get binned freqs for psychometrics
            inpt_data_all["binned_freq"] = pd.cut(inpt_data_all['stim'], bins=bin_lst,
                                                  labels=[str(x) for x in bin_name],
                                                  include_lowest=True)
            sim_stack = inpt_data_all.groupby(['binned_freq', 'state', 'animal'])['choice'].value_counts(
                normalize=True).unstack('choice').reset_index()
            sim_stack[0] = sim_stack[0].fillna(0)
            sim_stack['binned_freq'] = sim_stack['binned_freq'].astype('float')
            sim_stack = sim_stack.loc[(sim_stack.binned_freq > -0.7) & (sim_stack.binned_freq < 0.7)]

            # iterate over each state of K-state model
            for k_ind in range(K):
                ######################## K-STATE WEIGHTS ################################################
                new_ordered_weights_tmp = np.squeeze(global_weight_vectors[k_ind, :, :])
                bias_weight = new_ordered_weights_tmp[-1]
                new_ordered_weights_tmp = np.insert(new_ordered_weights_tmp, 0, bias_weight)
                ax[0, k_ind].plot(labels_for_plot_weights, new_ordered_weights_tmp[:-1], label='global',
                                  color='black', linewidth=1.5)
                ax[0, k_ind].set_xticklabels(labels_for_plot_weights, fontsize=12, rotation=45)
                ax[0, k_ind].axhline(0, linewidth=0.5, linestyle='--')
                ax[0, k_ind].set_title(state_label[k_ind])
                ax[0, k_ind].spines['right'].set_visible(False); ax[0, k_ind].spines['top'].set_visible(False)
                ######################## DWELL TIMES ####################################################
                # logbins = np.logspace(np.log10(1),
                #                       np.log10(max(state_dwell_times[:, k_ind])), 15)
                # # plot dwell times for each state
                # ax[2, k_ind].hist(state_dwell_times[:, k_ind],
                #                   bins=logbins,
                #                   color=cols[k_ind],
                #                   histtype='bar',
                #                   rwidth=0.8)
                # ax[2, k_ind].axvline(np.median(state_dwell_times[:, k_ind]),
                #                      linestyle='--',
                #                      color='k',
                #                      lw=1,
                #                      label='median')

                ##################### PLOT PSYCHOMETRICS FOR EACH K-STATE ######################################
                for animal in animal_list:
                    # x_data = inpt_data_all['stim'].loc[(inpt_data_all.stim > -0.7) & (inpt_data_all.stim < 0.7)
                    #     & (inpt_data_all.state == k_ind) & (inpt_data_all.animal == int(float(animal)))]
                    # y_data = -inpt_data_all['choice'].loc[(inpt_data_all.stim > -0.7) & (inpt_data_all.stim < 0.7)
                    #     & (inpt_data_all.state == k_ind) & (inpt_data_all.animal == int(float(animal)))]
                    # y_data = y_data.map({0:1, -1:-1})
                    # # y_data = y_data.map({1: 1, -1: 0})
                    # x_data = sim_stack['binned_freq'].loc[
                    #     (sim_stack.state == k_ind) & (sim_stack.animal == int(float(animal)))].unique()
                    # y_data = sim_stack[0].loc[
                    #     (sim_stack.state == k_ind) & (sim_stack.animal == int(float(animal)))]
                    # # p0 = [max(y_data), np.median(x_data), 1, min(y_data)]  # this is an mandatory initial guess
                    #
                    # try:
                    #     popt, pcov = curve_fit(sigmoid, x_data, y_data, method='dogbox')
                    # except RuntimeError:
                    #     print(animal + '_state_' + str(k_ind) + '_model_' + str(K))
                    #     continue
                    # y = sigmoid(x_data, *popt)
                    # ax[1, k_ind].plot(x_data, y, '--', color=cols[k_ind])

                    ax[1, k_ind].plot(sim_stack['binned_freq'].loc[
                        (sim_stack.state == k_ind) & (sim_stack.animal == int(float(animal)))].unique(), sim_stack[0].loc[
                        (sim_stack.state == k_ind) & (sim_stack.animal == int(float(animal)))], '--', color=cols[k_ind])
                ax[1, k_ind].plot(sim_stack.binned_freq.unique(),
                                  sim_stack[0].loc[(sim_stack.state == k_ind) & (sim_stack.animal == 0)],
                                  color='black')
                # miscellaneous
                ax[1, k_ind].get_xaxis().tick_bottom()
                ax[1, k_ind].set_xlim(-.65, .75)
                ax[1, k_ind].set_xticks(np.arange(-.7, .9, 0.2))
                ax[1, k_ind].set_xticklabels(labels=[str(np.round(x,2)) for x in np.arange(-.7, .9, 0.2)], fontsize=12,
                                              rotation=45)
                ax[1, k_ind].axhline(0.5, linewidth=0.5, linestyle='--')
                ax[1, k_ind].axvline(0, linewidth=0.5, linestyle='--')
                ax[1, k_ind].spines['right'].set_visible(False); ax[1, k_ind].spines['top'].set_visible(False)
            plt.tight_layout()
            plt.show()
            fig.savefig(save_folder / ('all_K' + str(K) + '_glmhmm_modelcheck_realdata.png'), format='png', bbox_inches="tight")
            del inpt_data_all



######################################################################################
##################### ALL ANIMALS ####################################################
##################### LOAD DATA ######################################################
if all_animals:
    # load dictionary for best cv model run
    with open(results_dir / "best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)
    # load cv array
    cv_file = results_dir / 'cvbt_folds_model.npz'
    cvbt_folds_model = load_cv_arr(cv_file)

    # load animal data for simulation
    inpt, y, session = load_data(data_dir / 'all_animals_concat.npz')
    inpt_unnorm, _, _ = load_data(data_dir / 'all_animals_concat_unnormalized.npz')
    # create dataframe all animals for plotting
    inpt_unnorm = np.append(inpt_unnorm, np.ones(inpt_unnorm.shape[0]).reshape(-1, 1), axis=1)
    inpt_data = pd.DataFrame(data=inpt_unnorm, columns=labels_for_plot)
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

    ## this is where recovery of global fits mismatch with individual fits, since the weights were neither called with
    ## get_global_weights nor permuted before refitting
    # get file name with best initialization given K value
    for K in range(2,K_max+1):
        raw_file = get_file_name_for_best_model_fold(
            cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
        hmm_params, lls = load_glmhmm_data(raw_file)
        weight_vectors = -hmm_params[2]
        log_transition_matrix = hmm_params[1][0]
        init_state_dist = hmm_params[0][0]

        # get posterior probs for state inference
        posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                                 hmm_params, K, range(K))
        states_max_posterior = np.argmax(posterior_probs, axis=1)
        inpt_data['state'] = states_max_posterior

        ##################### SIMULATE VEC #################################################
        # SIMULATE FROM GLMHMM
        # define col names for simulated dataframe
        col_names_glmhmm = labels_for_plot + ['choice','outcome','stim_org','state', 'session','y']
        # for global glmhmm model, multiple ses-sions simulation will not match the fitted model, since the fitted model used aggregated data
        n_session = n_session_lst[-1]
        n_trials = n_trials_lst[-1]
        # instantiate model
        data_hmm = ssm.HMM(K, D, M,
                               observations="input_driven_obs",
                               observation_kwargs=dict(C=2),
                               transitions="standard")
        data_hmm.params = hmm_params
        # initialize list of inputs and y
        glmhmm_inpt_lst, glmhmm_y_lst =[], []
        # initialize simulated array
        glmhmm_inpt_arr = np.zeros(len(col_names_glmhmm)).reshape(1,-1)
        for i in range(n_session):
            if i == n_session - 1:
                # for switching label problem
                n_trials = 5000
                # higher state needs more trials for permutation
                if K == 5:
                   n_trials = 10000
            # simulate stim vec
            stim_vec_sim = simulate_stim(n_trials + 1)
            # z score stim vec
            z_stim_sim = (stim_vec_sim - np.mean(stim_vec_sim)) / np.std(stim_vec_sim)
            # simulate data
            glmhmm_inpt, glmhmm_y, glmhmm_choice, glmhmm_outcome, glmhmm_state_arr = simulate_from_glmhmm_pfailpchoice_model(data_hmm,n_trials,z_stim_sim,stim_vec_sim,pfail)
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
        glmhmm_sim_df.to_csv('simulated_global_om_glmhmm_K'+str(K)+'.csv',index=False)
        # Calculate true loglikelihood
        true_ll = data_hmm.log_probability(glmhmm_y_lst, inputs=glmhmm_inpt_lst)
        print("true ll = " + str(true_ll))
        # fit glmhmm and perform recovery analysis
        recovered_glmhmm = ssm.HMM(K, D, M, observations="input_driven_obs",
                           observation_kwargs=dict(C=2), transitions="standard")
        N_iters = 200 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
        fit_ll = recovered_glmhmm.fit(glmhmm_y_lst, inputs=glmhmm_inpt_lst, method="em", num_iters=N_iters, tolerance=10**-4)
        # permute states
        # have to use the the last session which has huge n_trials beecause not all sessions have 4 states
        permute_df = glmhmm_sim_df.loc[(glmhmm_sim_df.session==n_session-1)]
        recovered_glmhmm.permute(find_permutation(permute_df.state.astype('int'), recovered_glmhmm.most_likely_states(np.array(permute_df.y).reshape(-1,1).astype('int'), input=np.array(permute_df[labels_for_plot]))))
        # initialize list of inputs and y
        glmhmm_inpt_recv_lst, glmhmm_y_recv_lst = [], []
        # initialize simulated array
        glmhmm_inpt_recv_arr = np.zeros(len(col_names_glmhmm)).reshape(1, -1)
        for i in range(n_session):
            if i == n_session - 1:
                # for switching label problem
                n_trials = 5000
            # simulate stim vec
            stim_vec_sim = simulate_stim(n_trials + 1)
            # z score stim vec
            z_stim_sim = (stim_vec_sim - np.mean(stim_vec_sim)) / np.std(stim_vec_sim)
            # simulate data
            glmhmm_inpt, glmhmm_y, glmhmm_choice, glmhmm_outcome, glmhmm_state_arr = simulate_from_glmhmm_pfailpchoice_model(
                recovered_glmhmm, n_trials, z_stim_sim,stim_vec_sim,pfail)
            # append list for model fit and recovery
            glmhmm_inpt_recv_lst.append(glmhmm_inpt)
            glmhmm_y_recv_lst.append(glmhmm_y)
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
            glmhmm_inpt_recv_arr = np.vstack((glmhmm_inpt_recv_arr, glmhmm_inpt))
        # create dataframe for plotting
        glmhmm_recv_df = pd.DataFrame(data=glmhmm_inpt_recv_arr[1:, :], columns=col_names_glmhmm)
        glmhmm_recv_df.to_csv('simulated_recv_global_om_glmhmm_K' + str(K) + '.csv', index=False)

        # CHECK PLOTS FOR SIMULATION AND RECOVERY
        # Plot the log probabilities of the true and fit models. Fit model final LL should be greater than or equal to true LL.
        fig = plt.figure(figsize=(10, 3.5), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(1, 3, 1)
        plt.plot(fit_ll, label="EM")
        plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ':k', label="True")
        plt.legend(loc="lower right")
        plt.xlim(0, len(fit_ll))
        plt.xlabel("EM Iteration", fontsize=12)
        plt.ylabel("Log Probability", fontsize=12)

        # plot generative and recovered transitrion matrix
        num_states = K
        plt.subplot(1, 3, 2)
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
        plt.ylabel("state t", fontsize=12)
        plt.xlabel("state t+1", fontsize=12)
        # plt.title("generative", fontsize = 15)

        plt.subplot(1, 3, 3)
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
        # plt.title("recovered", fontsize = 15)
        plt.ylabel("state t", fontsize=12)
        plt.xlabel("state t+1", fontsize=12)
        fig.suptitle('Generative vs recovered models', fontsize=15, y=0.98)
        fig.subplots_adjust(top=0.85);
        plt.show()
        fig.savefig(save_folder / ('global_K' + str(K) + '_simulated_data_model_fit.png'), format='png',
                    bbox_inches="tight")

        ##################### PSYCHOMETRIC CURVES ##########################################
        # since min/max freq_trans is -1.5/1.5
        bin_lst = np.arange(-1.55,1.6,0.1)
        bin_name=np.round(np.arange(-1.5,1.6,.1),2)
        # get binned freqs for psychometrics for simulated data
        glmhmm_sim_df["binned_freq"] = pd.cut(glmhmm_sim_df['stim_org'], bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)
        sim_stack = glmhmm_sim_df.groupby(['binned_freq','state'])['choice'].value_counts(normalize=True).unstack('choice').reset_index()
        sim_stack[-1] = sim_stack[-1].fillna(0)
        # get binned freqs for psychometrics for simulated recovered data
        glmhmm_recv_df["binned_freq"] = pd.cut(glmhmm_recv_df['stim_org'], bins=bin_lst,
                                              labels=[str(x) for x in bin_name], include_lowest=True)
        recv_stack = glmhmm_recv_df.groupby(['binned_freq', 'state'])['choice'].value_counts(normalize=True).unstack(
            'choice').reset_index()
        recv_stack[-1] = recv_stack[-1].fillna(0)
        # get binned freqs for psychometrics for real data
        inpt_data["binned_freq"] = pd.cut(inpt_data['stim'], bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)
        data_stack = inpt_data.groupby(['binned_freq','state'])['choice'].value_counts(normalize=True).unstack('choice').reset_index()
        data_stack[0] = data_stack[0].fillna(0)
        data_stack['binned_freq'] = data_stack['binned_freq'].astype('float')
        ##################### PLOT WEIGHTS FOR EACH K ######################################
        recovered_weights = -recovered_glmhmm.observations.params
        fig, ax= plt.subplots(2,K,figsize=(10,6),sharey="row",sharex='row')
        for ax_ind in range(K):
            # plot k-state weights
            ax[0,ax_ind].plot(labels_for_plot,np.squeeze(weight_vectors[ax_ind,:,:]),label='generated',color='blue')
            ax[0,ax_ind].plot(labels_for_plot, np.squeeze(recovered_weights[ax_ind, :, :]),'--',label='recovered',color='orange')
            ax[0,ax_ind].set_xticklabels(labels_for_plot, fontsize=12,rotation=45)
            ax[0,ax_ind].axhline(0,linewidth=0.5,linestyle='--')
            ax[0,ax_ind].set_title('state '+ str(ax_ind))
            # plot psychometrics for aggregated real data
            ax[1, ax_ind].plot(sim_stack.binned_freq.unique(), data_stack[0].loc[(data_stack.state == ax_ind)
                                                                                 & (data_stack.binned_freq > -0.7)
                                                                                 & (data_stack.binned_freq < 0.7)],label='data',color='black')
            # plot psychometrics for simulated data
            ax[1,ax_ind].plot(sim_stack.binned_freq.unique(),sim_stack[-1].loc[sim_stack.state==ax_ind], '--',label='generated',color='blue')
            # plot psychometrics for recovered data
            ax[1, ax_ind].plot(recv_stack.binned_freq.unique(), recv_stack[-1].loc[recv_stack.state == ax_ind], '--',
                               label='recovered',color='orange')
            # miscellaneous
            ax[1, ax_ind].set_xticklabels(labels= [str(x) for x in sim_stack.binned_freq.unique()] ,fontsize=12, rotation=45)
            ax[1, ax_ind].axhline(0.5, linewidth=0.5, linestyle='--')
            ax[1, ax_ind].axvline(6, linewidth=0.5, linestyle='--')
            if ax_ind == K-1:
                handles, labels = ax[1, ax_ind].get_legend_handles_labels()
                ax[1, ax_ind].legend(handles, labels, loc='lower right')
        fig.suptitle('All animals')
        plt.tight_layout()
        fig.savefig(save_folder / ('global_K' + str(K) + '_glmhmm_modelsimulations.png'), format='png', bbox_inches="tight")
        plt.show()

######################################################################################
##################### INDIVIDUAL ANIMAL ##############################################
##################### LOAD DATA ######################################################
if individual_animals:
    animal_lst = [str(x) for x in animal_list]
    for K in range(2,K_max+1):
        n_session = n_session_lst[-1]
        n_trials = n_trials_lst[-1]
        for animal in animal_lst:
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
            # Also get data for animal:
            inpt, y, session = load_data(data_individual / (animal + '_processed.npz'))
            inpt_unnorm, _, _ = load_data(data_individual / (animal + '_unnormalized.npz'))
            all_sessions = np.unique(session)
            # create dataframe single animals for plotting
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
            inpt_data['animal'] = np.ones(len(inpt_data))*int(float(animal))

            ##################### SIMULATE VEC #################################################
            # SIMULATE FROM GLMHMM
            # define col names for simulated dataframe
            col_names_glmhmm = labels_for_plot + ['choice','outcome','stim_org','state', 'session','y']
            # for global glmhmm model, multiple sessions simulation will not match the fitted model,
            # since the fitted model used aggregated data
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
                if i == n_session-1:
                    # for switching label problem
                    n_trials = 7500
                # simulate stim vec
                stim_vec_sim = simulate_stim(n_trials + 1)
                # z score stim vec
                z_stim_sim = (stim_vec_sim - np.mean(stim_vec_sim)) / np.std(stim_vec_sim)
                # simulate data
                glmhmm_inpt, glmhmm_y, glmhmm_choice, glmhmm_outcome,\
                    glmhmm_state_arr = simulate_from_glmhmm_pfailpchoice_model(data_hmm,n_trials,z_stim_sim,stim_vec_sim,pfail)
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
            glmhmm_sim_df.to_csv('simulated_'+ animal + '_om_glmhmm_K' + str(K) + '.csv', index=False)
            # Calculate true loglikelihood
            true_ll = data_hmm.log_probability(glmhmm_y_lst, inputs=glmhmm_inpt_lst)
            print("true ll = " + str(true_ll))
            # fit glmhmm and perform recovery analysis
            recovered_glmhmm = ssm.HMM(K, D, M, observations="input_driven_obs",
                               observation_kwargs=dict(C=2), transitions="standard")
            N_iters = 200 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
            fit_ll = recovered_glmhmm.fit(glmhmm_y_lst, inputs=glmhmm_inpt_lst, method="em", num_iters=N_iters, tolerance=10**-4)
            # permute states
            # have to use the last session which has huge n_trials beecause not all sessions have 4 states
            permute_df = glmhmm_sim_df.loc[(glmhmm_sim_df.session==n_session-1)]
            recovered_glmhmm.permute(
                find_permutation(permute_df.state.astype('int'), recovered_glmhmm.most_likely_states(np.array(permute_df.y).reshape(-1,1).astype('int'), input=np.array(permute_df[labels_for_plot]))))
            # initialize list of inputs and y
            glmhmm_inpt_recv_lst, glmhmm_y_recv_lst = [], []
            # initialize simulated array
            glmhmm_inpt_recv_arr = np.zeros(len(col_names_glmhmm)).reshape(1, -1)
            for i in range(n_session):
                if i == n_session - 1:
                    # for switching label problem
                    n_trials = 5000
                # simulate stim vec
                stim_vec_sim = simulate_stim(n_trials + 1)
                # z score stim vec
                z_stim_sim = (stim_vec_sim - np.mean(stim_vec_sim)) / np.std(stim_vec_sim)
                # simulate data
                glmhmm_inpt, glmhmm_y, glmhmm_choice, glmhmm_outcome, glmhmm_state_arr = simulate_from_glmhmm_pfailpchoice_model(
                    recovered_glmhmm, n_trials, z_stim_sim,stim_vec_sim,pfail)
                # append list for model fit and recovery
                glmhmm_inpt_recv_lst.append(glmhmm_inpt)
                glmhmm_y_recv_lst.append(glmhmm_y)
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
                glmhmm_inpt_recv_arr = np.vstack((glmhmm_inpt_recv_arr, glmhmm_inpt))
            # create dataframe for plotting
            glmhmm_recv_df = pd.DataFrame(data=glmhmm_inpt_recv_arr[1:, :], columns=col_names_glmhmm)
            glmhmm_recv_df.to_csv('simulated_recv_'+animal+'_om_glmhmm_K' + str(K) + '.csv', index=False)

            # CHECK PLOTS FOR SIMULATION AND RECOVERY
            # Plot the log probabilities of the true and fit models. Fit model final LL should be greater than or equal to true LL.
            fig = plt.figure(figsize=(10, 3.5), dpi=80, facecolor='w', edgecolor='k')
            plt.subplot(1, 3, 1)
            plt.plot(fit_ll, label="EM")
            plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ':k', label="True")
            plt.legend(loc="lower right")
            plt.xlim(0, len(fit_ll))
            plt.xlabel("EM Iteration", fontsize = 12)
            plt.ylabel("Log Probability", fontsize = 12)

            # plot generative and recovered transitrion matrix
            num_states=K
            plt.subplot(1, 3, 2)
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
            plt.ylabel("state t", fontsize = 12)
            plt.xlabel("state t+1", fontsize = 12)
            # plt.title("generative", fontsize = 15)

            plt.subplot(1, 3, 3)
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
            # plt.title("recovered", fontsize = 15)
            plt.ylabel("state t", fontsize = 12)
            plt.xlabel("state t+1", fontsize = 12)
            fig.suptitle('Generative vs recovered models - animal ' + animal,fontsize=15,y=0.98)
            fig.subplots_adjust(top=0.85);plt.show()
            fig.savefig(save_folder / (animal +'_K'+ str(K)+'_simulated_data_model_fit'+'.png'),format='png',bbox_inches = "tight")

            ##################### PSYCHOMETRIC CURVES ##########################################
            # since min/max freq_trans is -1.5/1.5
            bin_lst = np.arange(-1.55,1.6,0.1)
            bin_name=np.round(np.arange(-1.5,1.6,.1),2)
            # get binned freqs for psychometrics for simulated data
            glmhmm_sim_df["binned_freq"] = pd.cut(glmhmm_sim_df['stim_org'], bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)
            sim_stack = glmhmm_sim_df.groupby(['binned_freq','state'])['choice'].value_counts(normalize=True).unstack('choice').reset_index()
            sim_stack[-1] = sim_stack[-1].fillna(0)
            sim_stack['binned_freq'] = sim_stack['binned_freq'].astype('float')
            # get binned freqs for psychometrics for simulated recovered data
            glmhmm_recv_df["binned_freq"] = pd.cut(glmhmm_recv_df['stim_org'], bins=bin_lst,
                                                   labels=[str(x) for x in bin_name], include_lowest=True)
            recv_stack = glmhmm_recv_df.groupby(['binned_freq', 'state'])['choice'].value_counts(
                normalize=True).unstack(
                'choice').reset_index()
            recv_stack[-1] = recv_stack[-1].fillna(0)
            recv_stack['binned_freq'] = recv_stack['binned_freq'].astype('float')
            # get binned freqs for psychometrics for real data
            inpt_data["binned_freq"] = pd.cut(inpt_data['stim'], bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)
            data_stack = inpt_data.groupby(['binned_freq','state'])['choice'].value_counts(normalize=True).unstack('choice').reset_index()
            data_stack[0] = data_stack[0].fillna(0)
            data_stack['binned_freq'] = data_stack['binned_freq'].astype('float')
            # sns.lineplot(sim_stack.binned_freq,sim_stack[-1],hue=sim_stack.state);plt.show()

            ##################### PLOT WEIGHTS FOR EACH K ######################################
            recovered_weights = -recovered_glmhmm.observations.params
            fig, ax= plt.subplots(2,K,figsize=(10,6),sharey="row")
            for ax_ind in range(K):
                # plot k-state weights
                ax[0, ax_ind].plot(labels_for_plot, np.squeeze(weight_vectors[ax_ind, :, :]), label='generated',
                                   color='blue')
                ax[0, ax_ind].plot(labels_for_plot, np.squeeze(recovered_weights[ax_ind, :, :]), '--',
                                   label='recovered', color='orange')
                ax[0, ax_ind].set_xticklabels(labels_for_plot, fontsize=12, rotation=45)
                ax[0, ax_ind].axhline(0, linewidth=0.5, linestyle='--')
                ax[0, ax_ind].set_title('state ' + str(ax_ind))
                # plot psychometrics for aggregated real data
                ax[1, ax_ind].plot(data_stack.binned_freq.loc[(data_stack.state == ax_ind)].unique(), data_stack[0].loc[(data_stack.state == ax_ind)],
                                   label='data', color='black')
                # plot psychometrics for simulated data
                ax[1, ax_ind].plot(sim_stack.binned_freq.unique(), sim_stack[-1].loc[sim_stack.state == ax_ind], '--',
                                   label='generated', color='blue')
                # plot psychometrics for recovered data
                ax[1, ax_ind].plot(recv_stack.binned_freq.unique(), recv_stack[-1].loc[recv_stack.state == ax_ind],
                                   '--',
                                   label='recovered', color='orange')
                # miscellaneous
                ax[1, ax_ind].get_xaxis().tick_bottom()
                ax[1, ax_ind].set_xlim(-1, 1)
                ax[1, ax_ind].set_xticks(np.arange(-1,1.25,0.25))
                ax[1, ax_ind].set_xticklabels( labels=[str(x) for x in np.arange(-1,1.25,0.25)] ,fontsize=12, rotation=45)
                ax[1, ax_ind].axhline(0.5, linewidth=0.5, linestyle='--')
                ax[1, ax_ind].axvline(0, linewidth=0.5, linestyle='--')
                if ax_ind == K-1:
                    handles, labels = ax[1, ax_ind].get_legend_handles_labels()
                    ax[1, ax_ind].legend(handles, labels,loc='lower right')
            fig.suptitle('Animal '+ animal,fontsize=15,y=.98)
            fig.subplots_adjust(top=0.9, hspace=0.4);plt.show()
            fig.savefig(save_folder / (animal +'_K'+ str(K) + '_weights_psychometrics' + animal +'_testperm.png'),format='png',bbox_inches = "tight")

##############################################################################################
##################### PLOT POSTERIOR PROBS (ANIMAL SPECIFIC) #################################
if one_animal:
    animal_lst = [19.0, 5.0, 1.0]
    animal_lst = [str(x) for x in animal_lst]
    K = 3
    session_length = 200
    session_max = 250
    for animal in animal_list:

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

        # Also get data for animal:
        inpt, y, session = load_data(data_individual / (animal + '_processed.npz'))
        all_sessions = np.unique(session)
        # prepare data
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                       inpt.shape[0])
        y[np.where(y == -1), :] = 1
        inputs, datas, train_masks = partition_data_by_session(
            np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)
        # get posterior probs for state inference
        posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                                 hmm_params, K, range(K))
        states_max_posterior = np.argmax(posterior_probs, axis=1)
        posterior_probs_array = np.zeros((session_length,K))

        for i, sess in enumerate(all_sessions):
            # get session index from session array
            idx_session = np.where(session == sess)
            # get posterior probs according to session index
            posterior_probs_this_session = posterior_probs[idx_session[0], :]
            # if session longer than session_max, discard
            if len(idx_session[0]) > session_max:
                continue
            # if session shorter than session_length, discard
            try:
                posterior_probs_array = np.dstack([posterior_probs_array, posterior_probs_this_session[:session_length,:]])
            except:
                continue
        posterior_probs_array = posterior_probs_array[:,:,1:]

        fig, ax = plt.subplots(figsize=(10, 6))
        for k in range(K):
            tmp_state = posterior_probs_array[:,k, :]
            sem_tmp = sem(tmp_state, axis=1)
            ax.plot(np.arange(session_length), np.mean(tmp_state,axis=1), label="State " + str(k + 1), lw=1, color=cols[k])
            ax.fill_between(np.arange(session_length),
                            np.mean(tmp_state,axis=1) - sem_tmp,
                            np.mean(tmp_state,axis=1) + sem_tmp,alpha=0.2,color=cols[k])
        plt.ylim((-0.01, 1.01))
        plt.xlabel("trial #", fontsize=10)
        plt.ylabel("p(state)", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower right'); sns.despine(fig,top=True,right=True)
        fig.suptitle('State change dynamics ' + animal); plt.tight_layout()
        fig.savefig(save_folder / (animal + '_state_changedynamics_K' + str(K) + 'averagedsess.png'), format='png', bbox_inches="tight")

        #
        # sess_to_plot = [all_sessions[4],all_sessions[5],all_sessions[6],all_sessions[7],
        #                 all_sessions[9],all_sessions[10],all_sessions[11],all_sessions[12],
        #                 all_sessions[14],all_sessions[15],all_sessions[16],all_sessions[17],
        #                all_sessions[24],all_sessions[25],all_sessions[26],all_sessions[27],
        #                all_sessions[29],all_sessions[30],all_sessions[31],all_sessions[32]]
        # plt_row_ind = [1,2,3,4,5]
        # plt_sess_ind = [4,8,12,16,20]
        # fig,ax = plt.subplots(5,4,figsize=(10, 12),sharey='row')
        #
        # plt_row_index = 0
        # time_plot = 0
        # for i, sess in enumerate(sess_to_plot):
        #     time_plot += 1
        #     if time_plot == 4:
        #         time_plot = 0
        #     if i in plt_sess_ind:
        #         plt_row_index += 1
        #     # get session index from session array
        #     idx_session = np.where(session == sess)
        #     # get input according to the session index
        #     this_inpt = inpt[idx_session[0], :]
        #     # get posterior probs according to session index
        #     posterior_probs_this_session = posterior_probs[idx_session[0], :]
        #     # Plot trial structure for this session too:
        #     for k in range(K):
        #         ax[plt_row_index,time_plot-1].plot(posterior_probs_this_session[:, k],
        #                  label="State " + str(k + 1), lw=1,
        #                  color=cols[k])
        #     # get max probs of state of each trial
        #     states_this_sess = states_max_posterior[idx_session[0]]
        #     # get state change index
        #     state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
        #     # plot state change
        #     for change_loc in state_change_locs:
        #         ax[plt_row_index,time_plot-1].axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
        #     plt.ylim((-0.01, 1.01))
        #     plt.title("example session " + str(i + 1), fontsize=10)
        #     plt.gca().spines['right'].set_visible(False)
        #     plt.gca().spines['top'].set_visible(False)
        #     if i == 0:
        #         plt.xlabel("trial #", fontsize=10)
        #         plt.ylabel("p(state)", fontsize=10)
        #     if i == len(sess_to_plot)-1:
        #         handles, labels = ax[plt_row_index, time_plot-1].get_legend_handles_labels()
        #         ax[plt_row_index, time_plot-1].legend(handles, labels, loc='lower right')
        #     ax[plt_row_index, time_plot - 1].set_title(str(sess))
        # fig.suptitle('State change dynamics '+animal)
        # fig.subplots_adjust(top=0.92, hspace=0.4);
        # fig.savefig('plots/'+animal+'_state_changedynamics_K' + str(K) + '.png', format='png', bbox_inches="tight")
        # plt.show()

##############################################################################################
##################### EXPLORATORY PLOTS ######################################################
if exploratory_plot:
    # load dictionary for best cv model run
    # with open(results_dir / "best_init_cvbt_dict.json", 'r') as f:
    #     best_init_cvbt_dict = json.load(f)
    # # load cv array
    # cv_file = results_dir / 'cvbt_folds_model.npz'
    # cvbt_folds_model = load_cv_arr(cv_file)
    #

    # # since min/max freq_trans is -1.5/1.5
    bin_lst = np.arange(-1.55, 1.6, 0.1)
    bin_name = np.round(np.arange(-1.5, 1.6, .1), 2)
    # # load animal data for simulation
    # inpt, y, session = load_data(data_dir / 'all_animals_concat.npz')
    # inpt_unnorm, _, _ = load_data(data_dir / 'all_animals_concat_unnormalized.npz')
    # # create dataframe all animals for plotting
    # inpt_unnorm = np.append(inpt_unnorm, np.ones(inpt_unnorm.shape[0]).reshape(-1, 1), axis=1)
    # inpt_data = pd.DataFrame(data=inpt_unnorm, columns=labels_for_plot)
    # inpt_data['choice'] = y
    # # prepare data
    # violation_idx = np.where(y == -1)[0]
    # nonviolation_idx, mask = create_violation_mask(violation_idx,
    #                                                inpt.shape[0])
    # y[np.where(y == -1), :] = 1
    # inputs, datas, train_masks = partition_data_by_session(
    #     np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)
    #
    # raw_file = get_file_name_for_best_model_fold(
    #     cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    # hmm_params, lls = load_glmhmm_data(raw_file)
    # # since this is all animals and the original glmhmm params are not permuted
    # permutation = calculate_state_permutation(hmm_params)
    # # get posterior probs for state inference
    # # replacing range(K) with permutation helps align state of the global with individual glmhmm states' weights
    # posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
    #                                          hmm_params, K, permutation)
    # states_max_posterior = np.argmax(posterior_probs, axis=1)
    # # create state column
    # inpt_data['state'] = states_max_posterior
    # # get mouse and session id
    # inpt_data['session_info'] = session
    # inpt_data[['mouse_id','session_id']] = inpt_data['session_info'].str.split('s',expand=True)
    # inpt_data['mouse_id'] = np.zeros(len(inpt_data))
    # # transform choice (one-hot encoding) to convenient encoding to create success colum
    # inpt_data['choice_trans'] = inpt_data['choice'].map({1:1, 0:-1})
    # inpt_data['success'] = np.zeros(len(inpt_data))
    # inpt_data['success'] = np.where(inpt_data['choice_trans']*inpt_data['stim'] < 0, 1, 0)
    # # create stacked dataframe for PE plotting
    # data_stack = inpt_data.groupby(['mouse_id', 'state', 'pfail'])['success'].value_counts(normalize=True).unstack('success').reset_index()
    # # sns.pointplot(data=data_stack.loc[(data_stack.pfail==1)],x='state',y=1,hue='mouse_id');plt.show()
    # diff_stack = inpt_data.groupby(['mouse_id', 'state', 'pfail'])['success'].value_counts(normalize=True).unstack('success').diff().reset_index()
    # # sns.pointplot(data=diff_stack.loc[(diff_stack.pfail == 1)], x='state', y=1, hue='mouse_id');
    #
    get_opto = 1
    # this is to get other columns for analysis
    if get_opto:
        raw_df = pd.read_csv(os.path.join(data_dir,'opto_om_batch2_processed.csv'))
        raw_df = raw_df.loc[(raw_df.sound_diff < 37) & (raw_df.sound_diff > 31)].copy().reset_index(drop=True)
        save_name = 'opto_om_state_info.csv'
    else:
        raw_df = pd.read_csv(os.path.join(data_dir, 'om_all_batch1&2&3&4_processed.csv'))
        save_name = 'om_state_info.csv'
    # each animal concat
    inpt_data_all = pd.DataFrame()
    for animal in animal_list:
        print(animal)

        results_dir_individual_animal = results_dir_individual / animal
        cv_file = results_dir_individual_animal / "cvbt_folds_model.npz"
        cvbt_folds_model = load_cv_arr(cv_file)

        with open(results_dir_individual_animal / "best_init_cvbt_dict.json", 'r') as f:
            best_init_cvbt_dict = json.load(f)

        # prepare data for state inference
        inpt, y, session = load_data(data_individual / (animal + '_processed.npz'))
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                       inpt.shape[0])
        y[np.where(y == -1), :] = 1
        inputs, datas, train_masks = partition_data_by_session(
            np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

        # prepare data for dataframe
        inpt_unnorm, _, _ = load_data(data_individual / (animal + '_unnormalized.npz'))
        inpt_unnorm = np.append(inpt_unnorm, np.ones(inpt_unnorm.shape[0]).reshape(-1, 1), axis=1)
        inpt_data = pd.DataFrame(data=inpt_unnorm, columns=labels_for_plot)
        inpt_data['choice'] = y

        # add info columns
        inpt_data['mouse_id'] = np.ones(len(inpt_data)) * int(float(animal))
        # transform choice (one-hot encoding) to convenient encoding to create success colum
        inpt_data['choice_trans'] = inpt_data['choice'].map({1: 1, 0: -1})
        # inpt_data['success'] = np.zeros(len(inpt_data))
        # this assignment will fail if stim == 0
        # inpt_data['success'] = np.where(inpt_data['choice_trans'] * inpt_data['stim'] < 0, 1, 0)
        inpt_data['success'] = np.asarray(raw_df['success'].loc[raw_df['mouse_id'] == float(animal)]).astype(int)
        inpt_data['pfail'] = np.asarray(raw_df['prev_failure'].loc[raw_df['mouse_id'] == float(animal)]).astype(int)
        inpt_data['RT'] = np.asarray(raw_df['RT'].loc[raw_df['mouse_id'] == float(animal)])
        # or this inpt_data['session_arr'] = session
        inpt_data['session_identifier'] = np.asarray(
            raw_df['session_identifier'].loc[raw_df['mouse_id'] == float(animal)])
        inpt_data['pstim'] = np.asarray(
            raw_df['prev_freq_trans'].loc[raw_df['mouse_id'] == float(animal)])
        # get binned freqs for psychometrics
        if get_opto:
            inpt_data["opto_trial"] = np.asarray(
            raw_df['opto_trial'].loc[raw_df['mouse_id'] == float(animal)])
            inpt_data["phase"] = np.asarray(
                raw_df['phase'].loc[raw_df['mouse_id'] == float(animal)])
        else:
            inpt_data["binned_freq"] = pd.cut(inpt_data['stim'], bins=bin_lst, labels=[str(x) for x in bin_name],
                                              include_lowest=True)

        # get state inference
        posterior_probs_pd = pd.DataFrame()
        for K in [3,4]:
            # Get the file name corresponding to the best initialization for given K value
            raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                         results_dir_individual_animal,
                                                         best_init_cvbt_dict)
            hmm_params, lls = load_glmhmm_data(raw_file)

            # Save parameters for initializing individual fits
            weight_vectors = -hmm_params[2]
            log_transition_matrix = hmm_params[1][0]
            init_state_dist = hmm_params[0][0]

            # get posterior probs for state inference
            posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                                     hmm_params, K, range(K))
            states_max_posterior = np.argmax(posterior_probs, axis=1)
            # add state prob information
            posterior_probs_pd_tmp = pd.DataFrame(posterior_probs)
            dict_K_name = {0: 'K' + str(K) + '_state_0', 1: 'K' + str(K) + '_state_1', 2: 'K' + str(K) + '_state_2',
                           3: 'K' + str(K) + '_state_3'}
            dict_current_K = {k: dict_K_name[k] for k in sorted(dict_K_name.keys())[:K]}
            posterior_probs_pd_tmp.rename(columns=dict_current_K, inplace=True)
            # state with max prob
            posterior_probs_pd_tmp['K' + str(K) + '_state'] = states_max_posterior
            # location of state change
            posterior_probs_pd_tmp['K' + str(K) + '_state_change'] = (posterior_probs_pd_tmp['K' + str(K) + '_state']).diff()
            posterior_probs_pd_tmp.loc[
                posterior_probs_pd_tmp['K' + str(K) + '_state_change'] != 0, 'K' + str(K) + '_state_change'] = 1

            posterior_probs_pd = pd.concat((posterior_probs_pd,posterior_probs_pd_tmp),axis=1)

        # concat state inference to individual animal dataframe
        inpt_data = pd.concat((inpt_data,posterior_probs_pd),axis=1)
        # concat individual animal to all animals
        inpt_data_all = pd.concat([inpt_data_all,inpt_data],ignore_index=True)
        del inpt_data, posterior_probs_pd

    inpt_data_all.to_csv(os.path.join(data_dir, save_name))
    # inpt_data_all = pd.read_csv(os.path.join(data_dir,'om_state_info.csv'))
    # inpt_data_all['state_valid'] = np.ones(len(inpt_data_all))
    # inpt_data_all['state_valid_2'] = np.ones(len(inpt_data_all))
    # index_data = inpt_data_all.index
    # these animals are with non-sticky state changes
    # https://www.biorxiv.org/content/10.1101/2024.02.13.580224v1.full
    # try to see the nature of rapid state switching before eliminating it
    # animal_exclude = ['1','5','14','15','16','17','18','19','25']
    # animal_list = list(set(animal_list) - set(animal_exclude))
    # data_stack_all = inpt_data_all.groupby(['mouse_id', 'state', 'pfail'])['success'].value_counts(normalize=True).unstack('success').reset_index()
    # fig, ax = plt.subplots(2,1,figsize=(6, 10))
    # sns.pointplot(data=data_stack_all.loc[(data_stack_all.pfail==0)],x='state',y=1,hue='mouse_id',ax=ax[0],lw=0.5,linestyles='--',legend=0);
    # sns.pointplot(data=data_stack.loc[(data_stack.pfail==0)],x='state',y=1,ax=ax[0], color='k',legend=0);ax[0].legend([],[], frameon=False)
    # ax[0].set_title('pfail accuracy'); ax[0].set_xlabel(''); ax[0].set_ylabel('accuracy',fontsize=10)
    #
    # diff_stack_all = inpt_data_all.groupby(['mouse_id', 'state', 'pfail'])['success'].value_counts(normalize=True).unstack('success').diff().reset_index()
    # sns.pointplot(data=diff_stack_all.loc[(diff_stack_all.pfail == 1)], x='state', y=1, hue='mouse_id',ax=ax[1],lw=0.5,linestyles='--',legend=0);
    # sns.pointplot(data=diff_stack.loc[(diff_stack.pfail == 1)], x='state', y=1, ax=ax[1], color='k',legend=0);ax[1].legend([],[], frameon=False);
    # ax[1].set_title(r'$\Delta$ accuracy pfail-psuccess'); ax[1].set_ylabel(r'$\Delta$ accuracy', fontsize=10)
    # plt.tight_layout(); plt.show();
    # fig.savefig('plots/all_K' + str(K) + '_pfail_delta_acc.png', format='png', bbox_inches="tight")
    #
    # inpt_data_all['choice_congruent'] = (np.array(inpt_data_all['choice_trans']) == np.array(inpt_data_all['pchoice'])).astype('int')
    # inpt_data_all['pstim'] = inpt_data_all['stim'].shift(periods=1).fillna(0)
    # inpt_data_all['stim_congruent'] = (
    #             np.sign(np.array(inpt_data_all['stim'])) == np.sign(np.array(inpt_data_all['pstim']))).astype('int')
    # inpt_data_all['stim_choice_congruent'] = (
    #         np.sign(np.array(-inpt_data_all['choice_trans'])) == np.sign(np.array(inpt_data_all['pstim']))).astype('int')

    # inpt_stack = inpt_data_try.groupby(['binned_freq', 'state', 'pfail','mouse_id'])['choice'].value_counts(normalize=True).unstack(
    #     'choice').reset_index()
    # inpt_stack[0] = inpt_stack[0].fillna(0)
    # inpt_stack['binned_freq'] = inpt_stack['binned_freq'].astype('float')
    #
    # cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    # g = sns.FacetGrid(inpt_stack,col='state', hue='pfail',hue_order=[0,1],height=3.5, aspect=.95)
    # g.map_dataframe(sns.lineplot, x='binned_freq', y=0); plt.legend(labels=['psuccess', 'pfail']); plt.ylabel('P(choose high)')
    # plt.show()

    # correct/incorrect matrix for the last and first 10 trials of a state
    n_trials = 3
    # for matrix plot
    fig = plt.figure(figsize=(10,10))
    columns_label = [str(x) for x in np.arange(K)]
    plotz = K
    index_state_change_all_big = index_data[inpt_data_all.state_change==1][1:]
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
        index_state_change_all = np.array(list(set(index_state_change_all_big).intersection(set(index_data[inpt_data_all.state == set_K]))))
        # get index of the state before the state change
        index_state_end = index_state_change_all - 1
        # get state identity before state change
        state_before = inpt_data_all['state'].iloc[index_state_end]
        # plt.hist(state_before);
        # plt.show()
        # fig,ax = plt.subplots(1,2,figsize=(10,5))
        # ax[1].hist(state_before,color='k',alpha=0.6);
        # ax[1].set_title('counts of states'); ax[1].set_xlabel('state identity')
        # loop over states before state change
        for k_num in range(K):
            # because theres no state change if state before match current state, set dummy array for plotting
            if k_num == set_K:
                state_success_lst.append(np.zeros(n_trials*2))
                state_choice_lst.append(np.zeros(n_trials*2))
                mean_state_bef_lst.append(0)
                mean_state_aft_lst.append(0)
                continue
            # get index of where state before match k_num
            index_state_change_tmp = np.where(state_before==k_num)
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
                    ind_tmp_aft = index_state_change_all_big[ind_tmp_big_list+1][0]
                elif ind_tmp_big_list == len(index_state_change_all_big)-1:
                    ind_tmp_bef = index_state_change_all_big[ind_tmp_big_list-1][0]
                    ind_tmp_aft = len(inpt_data_all)-1
                else:
                    ind_tmp_bef = index_state_change_all_big[ind_tmp_big_list-1][0]
                    ind_tmp_aft = index_state_change_all_big[ind_tmp_big_list+1][0]
                # get length state before and after, if lower than n_trials discard
                len_state_bef = index_state_change[i] - ind_tmp_bef
                len_state_aft = ind_tmp_aft - index_state_change[i]
                if len_state_bef < n_trials or len_state_aft < n_trials:
                    if len_state_bef < n_trials:
                        discard_len.append(len_state_bef)
                        discard_list.append(index_state_change[i]-1)
                        inpt_data_all['state_valid_2'].iloc[index_state_change[i]-len_state_bef:index_state_change[i]] = 0
                    else:
                        discard_len.append(len_state_aft)
                        discard_list.append(index_state_change[i])
                        inpt_data_all['state_valid_2'].iloc[
                        index_state_change[i]:index_state_change[i]+len_state_aft] = 0
                    continue
                mean_state_bef.append(np.mean(inpt_data_all['success'].iloc[ind_tmp_bef:index_state_change[i]]))
                mean_state_aft.append(np.mean(inpt_data_all['success'].iloc[index_state_change[i]:ind_tmp_aft]))

                # state end mean and length
                # start and end of the state change with success
                state_start_array_success = np.vstack(
                                [state_start_array_success, inpt_data_all['success'].iloc[index_state_change[i]:index_state_change[i] + n_trials]])
                state_end_array_success = np.vstack(
                    [state_end_array_success, inpt_data_all['success'].iloc[index_state_change[i] - n_trials:index_state_change[i]]])
                # start and end of the state change with choice
                state_start_array_choice = np.vstack(
                    [state_start_array_choice,
                     inpt_data_all['choice'].iloc[index_state_change[i]:index_state_change[i] + n_trials]])
                state_end_array_choice = np.vstack(
                    [state_end_array_choice,
                     inpt_data_all['choice'].iloc[index_state_change[i] - n_trials:index_state_change[i]]])
            # list of states
            state_success_lst.append(np.hstack([state_end_array_success[1:, :],state_start_array_success[1:,:]]))
            state_choice_lst.append(np.hstack([state_end_array_choice[1:, :],state_start_array_choice[1:,:]]))
            mean_state_bef_lst.append(np.mean(np.array(mean_state_bef)))
            mean_state_aft_lst.append(np.mean(np.array(mean_state_aft)))
        for i in range(plotz):
            if i==set_K:
                # initialize ax
                ax = plt.subplot2grid((plotz, plotz), (i, set_K),fig=fig)
                # set color for this subgrid
                ax.set_facecolor("#580F41")
                ax.yaxis.set_ticklabels([])
                ax.xaxis.set_ticklabels([])
                if i == plotz-1:
                    ax.set_xlabel('to '+ columns_label[set_K])
                if set_K == 0:
                    ax.set_ylabel('from ' +columns_label[set_K],rotation=0,ha='right')
                # Show all ticks and label them with the respective list entries

            else:
                print(i,set_K)
                ax = plt.subplot2grid((plotz, plotz), (i,set_K),fig=fig)
                ax.plot(np.mean(state_success_lst[i], axis=0), color=cols[i], lw=1)
                ax.plot(np.mean(state_choice_lst[i], axis=0), color=cols[i], lw=0.8, linestyle='--')
                ax.axhline(y=mean_state_bef_lst[i], xmin=0, xmax=0.4, linestyle='--', lw=.6, color=cols[i])
                ax.axhline(y=mean_state_aft_lst[i], xmin=0.6, xmax=1, linestyle='--', lw=.6, color=cols[set_K])
                ax.axvline(x=n_trials, linestyle='--', lw=.5, color='k');
                ax.axhline(y=0.5, linestyle='--', lw=.5, color='k')
                if i == plotz-1:
                    ax.set_xlabel('to ' + columns_label[set_K])
                else:
                    ax.xaxis.set_ticklabels([])
                if set_K == 0:
                    ax.set_ylabel('from ' + columns_label[i],rotation=0,ha='right')
                else:
                    ax.yaxis.set_ticklabels([])
            ax.set_ylim([-0.01, 1.01])
            ax.set_xlim([-0.1, n_trials*2-.9])
    # fig.savefig('plots/others_to_all_K_4_matrix.png', format='png', bbox_inches="tight")
    plt.show()
    # plot discard states and their lengths
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].hist(discard_len, color='violet', alpha=0.6);
    ax[1].hist(inpt_data_all['state'].iloc[discard_list],color='k',alpha=0.6)
    ax[1].set_title('counts of discarded states'); ax[1].set_xlabel('state identity')
    ax[0].set_title('counts of discarded lengths'); ax[0].set_xlabel('state length')
    # now, define the ticks (i.e. locations where the labels will be plotted)
    xticks = [i for i in range(K)]
    xticks_length = [i+1 for i in range(K)]
    # also define the labels we'll use (note this MUST have the same size as `xticks`!)
    xtick_labels = [str(x) for x in range(K)]
    xtick_labels_length = [str(x) for x in xticks_length]
    # add the ticks and labels to the plot
    ax[1].set_xticks(xticks); ax[1].set_xticklabels(xtick_labels); ax[1].set_xlim([-0.1,3.1]);
    ax[0].set_xticks(xticks_length); ax[0].set_xticklabels(xtick_labels_length); ax[0].set_xlim([0.9, 4.1]);
    plt.show()
    print('Done')

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = pd.read_csv('/home/anh/Documents/phd/outcome_manip/data/om_glmm_choice_pred.csv')
    data = pd.read_csv('/home/anh/Documents/phd/outcome_manip/data/om_state_info.csv')

    # to segregate RTs?
    from sklearn.mixture import GaussianMixture

    samples = np.array(data.RT.loc[data.success == 0])
    mixture = GaussianMixture(n_components=2).fit(samples.reshape(-1, 1))
    pred = GaussianMixture.predict(mixture, samples.reshape(-1, 1))


    index = data.index
    data['mouse_batch'] = np.zeros(len(data))
    data['mouse_batch'].loc[data.mouse_id <= 10] = 1
    data['mouse_batch'] = data['mouse_batch'].astype('str')

    data['mouse_int'] = np.zeros(len(data))
    data['mouse_int'].loc[data.mouse_id % 2 == 0] = 1
    data['mouse_id'].loc[data.mouse_int == 1].unique()

    data['spout_side'] = data['sound_index'].copy()
    data['spout_side'].loc[data['spout_side'] == 0] = 2
    left_side = index[(data.mouse_id % 2 !=0) & (data.sound_index == 0)]
    right_side = index[(data.mouse_id % 2 != 0) & (data.sound_index == 1)]
    data.loc[left_side, 'spout_side'] = 1
    data.loc[right_side, 'spout_side'] = 2

    left_array = data['spout_side'].where(data['spout_side'] != 2, 0)
    # array of correct low frequency trials, if sound_index is a high_index, change the accuracy to 0 (1->0 for all animals)
    left_correct = data['success'].where(data['spout_side'] != 2, 0)
    # print(low_correct.unique())
    left_acc = left_correct.rolling(window=5, min_periods=3).mean() / left_array.rolling(
        window=5, min_periods=3).mean()

    # array of high frequency trials, if sound_index is a low index, change them to 0 (1->0)
    right_array = data['spout_side'].where(data['spout_side'] != 1, 0)
    # print(high_array.unique())
    # 2 labels high freq, but we only want arrays of 1s and 0s, change 2s to 1s
    right_array = right_array.where(right_array != 2, 1)
    # array of correct high frequency trials, if sound_index is a low_index, change the accuracy to 0 (1->0 for all animals)
    right_correct = data['success'].where(data['spout_side'] != 1, 0)
    # print(high_correct.unique())
    right_acc = right_correct.rolling(window=5, min_periods=3).mean() / right_array.rolling(
        window=5, min_periods=3).mean()

    rolling_bias = -(left_acc.fillna(0) - right_acc.fillna(0))
    bin_lst = np.arange(-1.0, 1.02, 0.2)
    bin_name = np.round(bin_lst, 2)
    data['binned_rb'] = pd.cut(data['rolling_bias'], bins=bin_lst, labels=[str(x) for x in bin_name],
                               include_lowest=True)
    data['binned_rb'] = data['binned_rb'].shift(periods=2)

    bin_lst = np.arange(-1.55, 1.6, 0.1)
    bin_name = np.round(np.arange(-1.5, 1.6, .1), 2)
    data["binned_freq"] = pd.cut(data['freq_trans'], bins=bin_lst, labels=[str(x) for x in bin_name],
                                 include_lowest=True)
    data['lick_side_trans'] = data['lick_side'].copy()
    data['lick_side_trans'].loc[data.lick_side==2]=0

    # break down difficulty level of the previous trial
    data['difficulty_freq'] = np.zeros(len(data))
    data.loc[(data.prev_freq_trans < 0.3) & (data.prev_freq_trans > -.3), 'difficulty_freq'] = 1
    # get columns to restack data and plot
    wanted_cols = ['binned_freq','difficulty_freq','prev_choice', 'prev_failure', 'lick_side_freq']
    # only get steady state performance
    psych_data = data.loc[(data['freq_trans'] <= 0.5) & (data['freq_trans'] >= -0.5),wanted_cols].copy()
    psych_data_melt = pd.melt(psych_data, id_vars=wanted_cols[:-1],
                              value_vars=wanted_cols[-1:])
    melt_cols = wanted_cols[:-1]
    melt_cols.append('variable')
    inpt_stack = psych_data_melt.groupby(melt_cols)['value'].value_counts(
        normalize=True).unstack('value').reset_index()

    # repeat with the all data stack
    wanted_cols_all = ['binned_freq', 'prev_choice', 'prev_failure', 'lick_side_freq']
    psych_data = data.loc[(data['freq_trans'] <= 0.5) & (data['freq_trans'] >= -0.5), wanted_cols_all].copy()
    psych_data_melt = pd.melt(psych_data, id_vars=wanted_cols_all[:-1],
                              value_vars=wanted_cols_all[-1:])
    melt_cols = wanted_cols_all[:-1]
    melt_cols.append('variable')
    inpt_stack_all = psych_data_melt.groupby(melt_cols)['value'].value_counts(
        normalize=True).unstack('value').reset_index()


    # repeat with the all data stack
    wanted_cols_all_all = ['binned_freq', 'prev_choice', 'lick_side_freq']
    psych_data = data.loc[(data['freq_trans'] <= 0.5) & (data['freq_trans'] >= -0.5), wanted_cols_all_all].copy()
    psych_data_melt = pd.melt(psych_data, id_vars=wanted_cols_all_all[:-1],
                              value_vars=wanted_cols_all_all[-1:])
    melt_cols = wanted_cols_all_all[:-1]
    melt_cols.append('variable')
    inpt_stack_all_all = psych_data_melt.groupby(melt_cols)['value'].value_counts(
        normalize=True).unstack('value').reset_index()

    # plot individual and all

    g = sns.FacetGrid(inpt_stack, row='pchoice', col='pfail', height=3.5, aspect=.95)
    g.map_dataframe(sns.lineplot, x='binned_freq', y=0); plt.legend()
    # plt.show()
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        if i < 2:
            # p_choice_ind = -1
            # diff_freq_ind = i

            # diff_freq_ind = 0
            # p_fail_ind = i

            p_fail_ind = i
            p_choice_ind = -1
        else:
            # p_choice_ind = 1
            # diff_freq_ind = i - 2

            # diff_freq_ind = 1
            # p_fail_ind = i-2

            p_fail_ind = i-2
            p_choice_ind = 1
        # sns.lineplot(data=inpt_stack_all.loc[(inpt_stack_all.difficulty_freq==diff_freq_ind)
        #                                      &(inpt_stack_all.prev_choice==p_choice_ind)], x='binned_freq', y=0,ax=ax,color='k');
        # sns.lineplot(data=inpt_stack_all.loc[(inpt_stack_all.difficulty_freq == diff_freq_ind)
        #                                      &(inpt_stack_all.prev_failure==p_fail_ind)], x='binned_freq', y=0,ax=ax,color='k');

        # sns.lineplot(data=inpt_stack_all.loc[(inpt_stack_all.pchoice == p_choice_ind)
                                             # & (inpt_stack_all.pfail == p_fail_ind)], x='binned_freq', y=0, ax=ax, color='k');
        sns.lineplot(data=inpt_stack_all.loc[(inpt_stack_all.pchoice == p_choice_ind)],
                     x='binned_freq', y=0, ax=ax, color='brown', linestyle='dashed');

    var_plot = 'pstim_choice_congruent'
    inpt_stack = \
    data.loc[(data.stim < .6) & (data.stim > -.6)].groupby(['binned_freq',var_plot, 'pfail'])[
        'choice'].value_counts(
        normalize=True).unstack('choice').reset_index()
    inpt_stack_all = \
        data.loc[(data.stim < .6) & (data.stim > -.6)].groupby(['binned_freq', 'pfail'])[
            'choice'].value_counts(
            normalize=True).unstack('choice').reset_index()
    g = sns.FacetGrid(inpt_stack, hue=var_plot, row='pfail', height=3.5, aspect=.95)
    g.map_dataframe(sns.lineplot, x='binned_freq', y=0);
    plt.legend()
    # plt.show()
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        sns.lineplot(data=inpt_stack_all.loc[(inpt_stack_all['pfail'] == i)],
                     x='binned_freq', y=0, ax=ax, color='brown', linestyle='dashed');
    plt.show()
    # probability of repeat!?
    var_plot = 'choice_congruent'
    inpt_stack = \
        data.loc[(data.stim < .6) & (data.stim > -.6)].groupby(['binned_freq', 'pfail', 'pchoice'])[
            'choice_congruent'].value_counts(
            normalize=True).unstack('choice_congruent').reset_index()
    g = sns.FacetGrid(inpt_stack, col='pfail', hue='pchoice', height=3.5, aspect=.95)
    g.map_dataframe(sns.lineplot, x='binned_freq', y=1);
    plt.legend()
    plt.show()
    # with diff freq
    var_plot = 'choice_congruent'
    inpt_stack = \
        data.loc[(data.stim < .6) & (data.stim > -.6)].groupby(['binned_freq', var_plot, 'difficulty_freq', 'pfail'])[
            'choice'].value_counts(
            normalize=True).unstack('choice').reset_index()
    g = sns.FacetGrid(inpt_stack, col='pfail', hue=var_plot, row='difficulty_freq', height=3.5, aspect=.95)
    g.map_dataframe(sns.lineplot, x='binned_freq', y=0);
    plt.legend()
    plt.show()
    # stats of pfail pdiff trials
    inpt_stack = \
        data.loc[(data.stim < .6) & (data.stim > -.6)].groupby(['binned_freq', 'difficulty_freq'])[
            'pfail'].value_counts(normalize=True).unstack('pfail').reset_index()
    sns.lineplot(data=inpt_stack, x='binned_freq', y=1, hue='difficulty_freq');
    plt.show()

    data = pd.read_csv(
        '/home/anh/Documents/glmhmm/om_glm_hmm/2_fit_models/all_models_fit/simulated_global_om_glmhmm_K4.csv')
    data['abs_stim'] = abs(data['stim_org'].copy())
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    for i in range(4):
        sns.violinplot(data=data.loc[data.state == i], x='outcome', y='abs_stim', hue='pfail', split=True, ax=ax[i])
    plt.show()
    sns.violinplot(data=data, x='outcome', y='abs_stim', hue='pfail', split=True);
    plt.show()


    # PLOTS for meeting
    ## pfail pchoice psych
    hue_plot = 'pchoice'
    y_label = 'p(High)'
    inpt_stack = \
        data.loc[(data.stim < .6) & (data.stim > -.6)].groupby(['binned_freq', 'pfail','difficulty_freq', hue_plot])[
            'choice'].value_counts(
            normalize=True).unstack('choice').reset_index()
    inpt_stack_all = \
        data.loc[(data.stim < .6) & (data.stim > -.6)].groupby(['binned_freq', 'pfail'])[
            'choice'].value_counts(
            normalize=True).unstack('choice').reset_index()
    g = sns.FacetGrid(inpt_stack, col="pfail", row="difficulty_freq",hue=hue_plot, height=3.5, aspect=.8)
    g.map_dataframe(sns.lineplot, x='binned_freq', y=0);
    g.set(ylim=(0, 1));
    axes = g.axes.flatten()
    # iterate through the axes
    for i, ax in enumerate(axes):
        ax.axhline(0.5, ls='--', c='black', linewidth=0.2)
        ax.axvline(0, ls='--', c='black', linewidth=0.2)
        # sns.lineplot(data=inpt_stack_all.loc[inpt_stack_all.pfail==i], x='binned_freq',y=0,color='k',ax=ax,legend=False)
    g.set_axis_labels("Distance from threshold", "P(high)");
    plt.legend(title=hue_plot)
    g.tight_layout();
    plt.show(); g.savefig(save_folder+'/all_psych_pfail_pchoice_pdiff.png',format='png',bbox_inches = "tight")

    # congruent psych
    hue_plot = 'pfail'
    y_label = 'p(Repeat)'
    inpt_stack = \
        data.loc[(data.stim < .6) & (data.stim > -.6)].groupby(['binned_freq', 'pfail','pchoice'])[
            'choice_congruent'].value_counts(
            normalize=True).unstack('choice_congruent').reset_index()
    inpt_stack_all = \
        data.loc[(data.stim < .6) & (data.stim > -.6)].groupby(['binned_freq', 'pfail'])[
            'choice'].value_counts(
            normalize=True).unstack('choice').reset_index()
    g = sns.FacetGrid(inpt_stack, hue="pfail",col='pchoice', height=3.5, aspect=.85)
    g.map_dataframe(sns.lineplot, x='binned_freq', y=1);
    # g.set(ylim=(0, 1));
    axes = g.axes.flatten()
    # iterate through the axes
    for i, ax in enumerate(axes):
        # ax.axhline(0.5, ls='--', c='black', linewidth=0.2)
        ax.axvline(0, ls='--', c='black', linewidth=0.2)
        # sns.lineplot(data=inpt_stack_all.loc[inpt_stack_all.pfail==i], x='binned_freq',y=0,color='k',ax=ax,legend=False)
    g.set_axis_labels("Distance from threshold", "P(Repeat)");
    plt.legend(title=hue_plot)
    g.tight_layout();
    plt.show();
    g.savefig(save_folder + '/all_psych_choicecong_pfail_pchoice.png', format='png', bbox_inches="tight")

    data['choice_trans'] = data['lick_side_freq'].copy()
    data.loc[data.choice_trans==0, 'choice_trans'] = -1

    data['choice_congruent'] = data.choice_trans*data.prev_choice
    data.loc[data.sound_index==0,'sound_index'] = -1
    data['repeating_stim_evidence_raw'] = data.prev_choice*data.sound_index
    data['repeating_stim_evidence'] = data.repeating_stim_evidence_raw*data.abs_freq
    data['binned_freq_repeating'] = pd.cut(data['repeating_stim_evidence'], bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)
    data['d_freqs'] = data['abs_freq'] - abs(data['prev_freq_trans'])
    data['repeating_stim_evidence_d'] = data.repeating_stim_evidence_raw*data.d_freqs
    data['binned_freq_repeating_d']= pd.cut(data['repeating_stim_evidence_d'], bins=bin_lst, labels= [str(x) for x in bin_name], include_lowest=True)

    data['z_binned_freq'] = pd.cut(data['z_freq_trans'], bins=bin_lst, labels=[str(x) for x in bin_name],
                                   include_lowest=True)
    inpt_stack = \
        data.groupby(['mouse_id', 'binned_freq', 'prev_failure', 'prev_choice'])[
            'lick_side_freq'].value_counts(
            normalize=True).unstack('lick_side_freq').reset_index()
    g = sns.FacetGrid(inpt_stack, col="prev_failure", hue='prev_choice', height=3.5, aspect=.85)
    g.map_dataframe(sns.lineplot, x='binned_freq', y=0);
    g.map_dataframe(sns.scatterplot, x='binned_freq', y=0, s=10, legend=False);
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        ax.set_ylim([0, 1]);
        ax.set_xlim([-.7, .7])
        ax.axhline(0.5, ls='--', c='black', linewidth=0.2)
        ax.axvline(0, ls='--', c='black', linewidth=0.2)
    plt.show()
# TODO:
# figure out state transition dynamics
# https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011430
# -> model fit sanity checks (predictive accuracy, all weights all animals plots, dwell time, etc.)
# randomize initial condition for simulation?
# compare simulated and real data, see what the model captures in the data and what's lacking
# understand the state change dynamics
# simulation from glmhmm don't seem to look right????
# get psychometrics for each state and fraction correct, and fraction occupation
## figure 5def in the papser
## get posterior prob for each trial, get the psychometric accordingly and compare to simulated data from model each k
# systematically plot these diagnostic plots to understand the states
#