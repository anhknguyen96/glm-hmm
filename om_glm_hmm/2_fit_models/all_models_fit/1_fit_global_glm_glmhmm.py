#  Fit GLM to all IBL data together

import autograd.numpy as np
import autograd.numpy.random as npr
import os
import json
import sys
from glm_utils import load_session_fold_lookup, load_data, fit_glm, \
    plot_input_vectors, append_zeros
from pathlib import Path
from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
    load_data, create_violation_mask, launch_glm_hmm_job
from post_processing_utils import load_data, load_session_fold_lookup, \
    prepare_data_for_cv, calculate_baseline_test_ll, \
    calculate_glm_test_loglikelihood, calculate_cv_bit_trial, \
    return_glmhmm_nll, return_lapse_nll
from post_processing_utils import load_glmhmm_data, load_cv_arr, \
    create_cv_frame_for_plotting, get_file_name_for_best_model_fold, \
    permute_transition_matrix, calculate_state_permutation

# for global glm fit
C = 2  # number of output types/categories
N_initializations = 10
npr.seed(65)  # set seed in case of randomization

# for global glm-hmm fit
D = 1  # data (observations) dimension
N_em_iters = 300  # number of EM iterations
global_fit = True
transition_alpha = 1
prior_sigma = 100

# for global glm-hmm post processing
K_max = 5  # maximum number of latent states
num_models = K_max  # model for each latent
models_list = ["GLM", "GLM_HMM"]
def create_cluster_job(N_initializations,K_vals,num_folds,data_dir):
    cluster_job_arr = []
    for K in K_vals:
        for i in range(num_folds):
            for j in range(N_initializations):
                cluster_job_arr.append([K, i, j])
    np.savez(data_dir / "cluster_job_arr.npz",
             cluster_job_arr)

if __name__ == '__main__':
    # if len(sys.argv)==1:
    #     print('Please specify the data folder you want')
    #     exit()
    # root_folder_dir = str(sys.argv[1])
    root_folder_dir = '/home/anh/Documents/phd'

    root_folder_name = 'om_choice'
    root_data_dir = Path(root_folder_dir) / root_folder_name / 'data'
    root_result_dir = Path(root_folder_dir) / root_folder_name / 'result'
    # root_data_dir = Path('../../data')
    # root_result_dir = Path('../../results')

    data_dir = root_data_dir / (root_folder_name +'_data_for_cluster')
    # Create directory for results:
    results_dir = root_result_dir / (root_folder_name +'_global_fit')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    num_folds = 5
    K_vals = [2, 3, 4, 5]

    # Create cluster jobs to run global glm-hmm fit later
    create_cluster_job(N_initializations=20,K_vals=K_vals,num_folds=num_folds,data_dir=data_dir)

    # Subset to relevant covariates for covar set of interest:
    if root_folder_name == 'om_accuracy':
        labels_for_plot = ['prev_failure', 'sound_side', 'stim','intercept']
    else:
        labels_for_plot = ['prev-fail','stim', 'stim:prev-fail', 'prev-choice', 'bias']

    animal_file_name = 'all_animals_concat.npz'
    animal_file = data_dir / animal_file_name
    inpt, y, session = load_data(animal_file)
    session_fold_lookup_table = load_session_fold_lookup(
        data_dir / 'all_animals_concat_session_fold_lookup.npz')

    ###################################################################################
    # FIT GLOBAL GLM
    for fold in range(num_folds):

        # y = y.astype('int')
        figure_directory = results_dir / "GLM" / ("fold_" + str(fold))
        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        # Subset to sessions of interest for fold
        sessions_to_keep = session_fold_lookup_table[np.where(
            session_fold_lookup_table[:, 1] != fold), 0]
        idx_this_fold = [
            str(sess) in sessions_to_keep and y[id, 0] != -1
            for id, sess in enumerate(session)
        ]
        this_inpt, this_y, this_session = inpt[idx_this_fold, :], y[
            idx_this_fold, :], session[idx_this_fold]
        assert len(
            np.unique(this_y)
        ) == 2, "choice vector should only include 2 possible values"
        train_size = this_inpt.shape[0]

        M = this_inpt.shape[1]
        loglikelihood_train_vector = []

        for iter in range(N_initializations):  # GLM fitting should be
            # independent of initialization, so fitting multiple
            # initializations is a good way to check that everything is
            # working correctly
            loglikelihood_train, recovered_weights = fit_glm([this_inpt],
                                                             [this_y], M, C)
            weights_for_plotting = append_zeros(recovered_weights)
            plot_input_vectors(weights_for_plotting,
                               figure_directory,
                               title="GLM fit; Final LL = " +
                               str(loglikelihood_train),
                               save_title='init' + str(iter),
                               labels_for_plot=labels_for_plot)
            loglikelihood_train_vector.append(loglikelihood_train)
            np.savez(
                figure_directory / ('variables_of_interest_iter_' + str(iter) +
                '.npz'), loglikelihood_train, recovered_weights)

    ###################################################################################
    # FIT GLOBAL GLM-HMM
    # Load external files:
    cluster_arr_file = data_dir / 'cluster_job_arr.npz'
    # Load cluster array job parameters:
    cluster_arr = load_cluster_arr(cluster_arr_file)

    #  append a column of ones to inpt to represent the bias covariate:
    # we did not do that in fitting global glm because the glm function already had this in place
    inpt_mod = np.hstack((inpt, np.ones((len(inpt), 1))))
    y = y.astype('int')
    # Identify violations for exclusion:
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                   inpt.shape[0])

    for z in np.arange(cluster_arr.shape[0]):
        [K, fold, iter] = cluster_arr[z]
        #  GLM weights to use to initialize GLM-HMM
        init_param_file = results_dir / 'GLM' / ('fold_' + str(
            fold)) / 'variables_of_interest_iter_0.npz'

        # create save directory for this initialization/fold combination:
        save_directory = results_dir / ('GLM_HMM_K_' + str(
            K)) / ('fold_' + str(fold)) / ('iter_' + str(iter))
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        launch_glm_hmm_job(inpt_mod,
                           y,
                           session,
                           mask,
                           session_fold_lookup_table,
                           K,
                           D,
                           C,
                           N_em_iters,
                           transition_alpha,
                           prior_sigma,
                           fold,
                           iter,
                           global_fit,
                           init_param_file,
                           save_directory)

    # ###################################################################################
    # RUN GLOBAL GLM-HMM POST PROCESSING
    cvbt_folds_model = np.zeros((num_models, num_folds))
    cvbt_train_folds_model = np.zeros((num_models, num_folds))

    # Save best initialization for each model-fold combination
    best_init_cvbt_dict = {}
    for fold in range(num_folds):
        test_inpt, test_y, test_nonviolation_mask, this_test_session, \
            train_inpt, train_y, train_nonviolation_mask, this_train_session, M, \
            n_test, n_train = prepare_data_for_cv(
            inpt, y, session, session_fold_lookup_table, fold)
        ll0 = calculate_baseline_test_ll(
            train_y[train_nonviolation_mask == 1, :],
            test_y[test_nonviolation_mask == 1, :], C)
        ll0_train = calculate_baseline_test_ll(
            train_y[train_nonviolation_mask == 1, :],
            train_y[train_nonviolation_mask == 1, :], C)
        for model in models_list:
            print("model = " + str(model))
            if model == "GLM":
                # Load parameters and instantiate a new GLM object with
                # these parameters
                glm_weights_file = results_dir / 'GLM' / ('fold_' + str(
                    fold)) / 'variables_of_interest_iter_0.npz'
                ll_glm = calculate_glm_test_loglikelihood(
                    glm_weights_file, test_y[test_nonviolation_mask == 1, :],
                    test_inpt[test_nonviolation_mask == 1, :], M, C)
                ll_glm_train = calculate_glm_test_loglikelihood(
                    glm_weights_file, train_y[train_nonviolation_mask == 1, :],
                    train_inpt[train_nonviolation_mask == 1, :], M, C)
                cvbt_folds_model[0, fold] = calculate_cv_bit_trial(
                    ll_glm, ll0, n_test)
                cvbt_train_folds_model[0, fold] = calculate_cv_bit_trial(
                    ll_glm_train, ll0_train, n_train)
            elif model == "GLM_HMM":
                for K in range(2, K_max + 1):
                    print("K = " + str(K))
                    # the first index is GLM, next are the K-states GLM-hmm
                    model_idx = K-1
                    cvbt_folds_model[model_idx, fold], \
                        cvbt_train_folds_model[
                            model_idx, fold], _, _, init_ordering_by_train = \
                        return_glmhmm_nll(
                            np.hstack((inpt, np.ones((len(inpt), 1)))), y,
                            session, session_fold_lookup_table, fold,
                            K, D, C, results_dir)
                    # Save best initialization to dictionary for later:
                    key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(
                        fold)
                    best_init_cvbt_dict[key_for_dict] = int(
                        init_ordering_by_train[0])
    # Save best initialization directories across animals, folds and models
    # (only GLM-HMM):
    print(cvbt_folds_model)
    print(cvbt_train_folds_model)
    json_dump = json.dumps(best_init_cvbt_dict)
    f = open(results_dir / "best_init_cvbt_dict.json", "w")
    f.write(json_dump)
    f.close()
    # Save cvbt_folds_model as numpy array for easy parsing across all
    # models and folds
    np.savez(results_dir / "cvbt_folds_model.npz", cvbt_folds_model)
    np.savez(results_dir / "cvbt_train_folds_model.npz",
             cvbt_train_folds_model)

    ###################################################################################
    # GET BEST PARAMS FOR INDIVIDUAL INITIALIZATION
    save_directory = data_dir / "best_global_params"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for K in range(2, K_max + 1):
        print("K = " + str(K))
        with open(results_dir / "best_init_cvbt_dict.json", 'r') as f:
            best_init_cvbt_dict = json.load(f)

        # Get the file name corresponding to the best initialization for
        # given K value
        raw_file = get_file_name_for_best_model_fold(
            cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
        hmm_params, lls = load_glmhmm_data(raw_file)

        # Calculate permutation
        permutation = calculate_state_permutation(hmm_params)
        print(permutation)

        # Save parameters for initializing individual fits
        weight_vectors = hmm_params[2][permutation]
        log_transition_matrix = permute_transition_matrix(
            hmm_params[1][0], permutation)
        init_state_dist = hmm_params[0][0][permutation]
        params_for_individual_initialization = [[init_state_dist],
                                                [log_transition_matrix],
                                                weight_vectors]

        np.savez(
            save_directory / ('best_params_K_' + str(K) + '.npz'),
            params_for_individual_initialization)