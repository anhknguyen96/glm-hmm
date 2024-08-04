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
    create_violation_mask, get_marginal_posterior, get_global_weights, get_global_trans_mat, load_animal_list,\
    load_correct_incorrect_mat
from simulate_data_from_glm import *
from ssm.util import find_permutation


######################### PARAMS ####################################################
K_max = 5
root_folder_dir = '/home/anh/Documents'
root_folder_name = 'om_choice'
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

trouble_animals =['23.0','24.0','26.0']
animal_list = list(set(animal_list)-set(trouble_animals))

# flag for running all_animals analysis
all_animals = 0
# flag for running individual animals analysis
individual_animals = 0
# flag for running k-state glm fit check
glm_fit_check = 0
# flag for predictive accuracy plot
pred_acc_plot = 0
# flag for one animal
one_animal = 0

exploratory_plot = 1
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
    fig.savefig('plots/fig4_a'+'K_'+str(K_plot)+'_all.png',format='png', bbox_inches="tight")
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
    fig.savefig('plots/fig4_b'+'K_'+str(K_plot)+'_all.png',format='png', bbox_inches="tight")

##################### K-STATE GLM FIT CHECK ########################################
##################### SIMULATE VEC #################################################
if glm_fit_check:
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
            inpt_sim_tmp = simulate_from_weights_pfailpchoice_model(np.squeeze(global_weight_vectors[k_ind,:,:]),n_trials,z_stim_sim)
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
                                                                        z_stim_sim)
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
        fig.savefig('plots/all_K'+ str(K) + '_glmhmm_modelcheck.png',format='png',bbox_inches = "tight")

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
                recovered_glmhmm, n_trials, z_stim_sim)
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
        fig.savefig('plots/'+ 'global_K' + str(K) + '_simulated_data_model_fit.png', format='png',
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
        fig.savefig('plots/global_K' + str(K) + '_glmhmm_modelsimulations.png', format='png', bbox_inches="tight")
        plt.show()

######################################################################################
##################### INDIVIDUAL ANIMAL ##############################################
##################### LOAD DATA ######################################################
if individual_animals:
    animal_lst= [17.0,27.0,12.0]
    animal_lst = [str(x) for x in animal_lst]
    for K in [4]:
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
                    glmhmm_state_arr = simulate_from_glmhmm_pfailpchoice_model(data_hmm,n_trials,z_stim_sim)
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
                    recovered_glmhmm, n_trials, z_stim_sim)
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
            fig.savefig('plots/' + animal +'_K'+ str(K)+'_simulated_data_model_fit'+'.png',format='png',bbox_inches = "tight")

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
            fig.savefig('plots/' + animal +'_K'+ str(K) + '_weights_psychometrics' + animal +'_testperm.png',format='png',bbox_inches = "tight")

##############################################################################################
##################### PLOT POSTERIOR PROBS (ANIMAL SPECIFIC) #################################
if one_animal:
    animal_lst = [17.0, 27.0, 12.0]
    animal_lst = [str(x) for x in animal_lst]
    K = 4
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

        sess_to_plot = [all_sessions[4],all_sessions[5],all_sessions[6],all_sessions[7],
                        all_sessions[9],all_sessions[10],all_sessions[11],all_sessions[12],
                        all_sessions[14],all_sessions[15],all_sessions[16],all_sessions[17],
                       all_sessions[24],all_sessions[25],all_sessions[26],all_sessions[27],
                       all_sessions[29],all_sessions[30],all_sessions[31],all_sessions[32]]
        plt_row_ind = [1,2,3,4,5]
        plt_sess_ind = [4,8,12,16,20]

        fig,ax = plt.subplots(5,4,figsize=(10, 12),sharey='row')
        # plt.subplots_adjust(wspace=0.2, hspace=0.9)
        # for i, sess in enumerate(sess_to_plot):
        #     # plt.subplot(2, K, i+1)
        #     ax[0,i].plot(labels_for_plot, np.squeeze(weight_vectors[i, :, :]))
        #     ax[0,i].set_xticklabels(labels_for_plot, fontsize=12, rotation=45)
        #     ax[0,i].axhline(0, linewidth=0.5, linestyle='--')

        plt_row_index = 0
        time_plot = 0
        for i, sess in enumerate(sess_to_plot):
            time_plot += 1
            if time_plot == 4:
                time_plot = 0
            if i in plt_sess_ind:
                plt_row_index += 1
            # get session index from session array
            idx_session = np.where(session == sess)
            # get input according to the session index
            this_inpt = inpt[idx_session[0], :]
            # get posterior probs according to session index
            posterior_probs_this_session = posterior_probs[idx_session[0], :]
            # Plot trial structure for this session too:
            for k in range(K):
                ax[plt_row_index,time_plot-1].plot(posterior_probs_this_session[:, k],
                         label="State " + str(k + 1), lw=1,
                         color=cols[k])
            # get max probs of state of each trial
            states_this_sess = states_max_posterior[idx_session[0]]
            # get state change index
            state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
            # plot state change
            for change_loc in state_change_locs:
                ax[plt_row_index,time_plot-1].axvline(x=change_loc, color='k', lw=0.5, linestyle='--')
            plt.ylim((-0.01, 1.01))
            plt.title("example session " + str(i + 1), fontsize=10)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            if i == 0:
                plt.xlabel("trial #", fontsize=10)
                plt.ylabel("p(state)", fontsize=10)
            if i == len(sess_to_plot)-1:
                handles, labels = ax[plt_row_index, time_plot-1].get_legend_handles_labels()
                ax[plt_row_index, time_plot-1].legend(handles, labels, loc='lower right')
            ax[plt_row_index, time_plot - 1].set_title(str(sess))
        fig.suptitle('State change dynamics '+animal)
        fig.subplots_adjust(top=0.92, hspace=0.4);
        fig.savefig('plots/'+animal+'_state_changedynamics_K' + str(K) + '.png', format='png', bbox_inches="tight")
        plt.show()

##############################################################################################
##################### EXPLORATORY PLOTS ######################################################
if exploratory_plot:
    # load dictionary for best cv model run
    with open(results_dir / "best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)
    # load cv array
    cv_file = results_dir / 'cvbt_folds_model.npz'
    cvbt_folds_model = load_cv_arr(cv_file)

    K = 4
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

    raw_file = get_file_name_for_best_model_fold(
        cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)
    # get posterior probs for state inference
    posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                             hmm_params, K, range(K))
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    # create state column
    inpt_data['state'] = states_max_posterior
    # get mouse and session id
    inpt_data['session_info'] = session
    inpt_data[['mouse_id','session_id']] = inpt_data['session_info'].str.split('s',expand=True)
    inpt_data['mouse_id'] = inpt_data['mouse_id'].str.slice(start=1)
    # transform choice (one-hot encoding) to convenient encoding to create success colum
    inpt_data['choice_trans'] = inpt_data['choice'].map({1:1, 0:-1})
    inpt_data['success'] = np.zeros(len(inpt_data))
    inpt_data['success'] = np.where(inpt_data['choice_trans']*inpt_data['stim'] < 0, 1, 0)
    # create stacked dataframe for PE plotting
    data_stack = inpt_data.groupby(['mouse_id', 'state', 'pfail'])['success'].value_counts(normalize=True).unstack('success').reset_index()
    # TODO: which state is which?????
    sns.pointplot(data=data_stack.loc[(data_stack.pfail==1)],x='state',y=1,hue='mouse_id');plt.show()



# TODO:
# if animal == "ibl_witten_05" or animal == "CSHL_001":
#     permutation = calculate_state_permutation(hmm_params)
# else:
#     permutation = range(K)

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