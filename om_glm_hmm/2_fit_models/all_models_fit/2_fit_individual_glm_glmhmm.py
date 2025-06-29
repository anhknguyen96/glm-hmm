# Fit GLM to each IBL animal separately
import autograd.numpy as np
import autograd.numpy.random as npr
import os
import multiprocessing as mp
import json
from pathlib import Path
from glm_utils import load_session_fold_lookup, load_data, load_animal_list, \
    fit_glm, plot_input_vectors, append_zeros
from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
            load_animal_list, load_data, create_violation_mask, \
            launch_glm_hmm_job
from glm_hmm_utils import load_animal_list
from post_processing_utils import load_data, load_session_fold_lookup, \
    prepare_data_for_cv, calculate_baseline_test_ll, \
    calculate_glm_test_loglikelihood, calculate_cv_bit_trial, \
    return_glmhmm_nll, return_lapse_nll
npr.seed(65)

###################################################################################
# PARAMETERS
# individual glm fit
C = 2  # number of output types/categories
N_initializations = 10
num_folds = 5

# create cluster job individual glmhmm fit
prior_sigma = [2]
transition_alpha = [2]
K_vals = [2, 3, 4, 5]
N_em_iters = 300  # number of EM iterations
n_processes = 15
multiprocess_option=1

# post processing individual glmhmm fit
D = 1  # number of output dimensions
K_max = 5  # number of latent states
num_models = K_max # model for each latent
models_list = ["GLM", "GLM_HMM"]

def create_cluster_job_individual_fit(N_initializations,K_vals,num_folds,prior_sigma,transition_alpha,data_dir):
    cluster_job_arr = []
    for K in K_vals:
        for i in range(num_folds):
            for j in range(N_initializations):
                for sigma in prior_sigma:
                    for alpha in transition_alpha:
                        cluster_job_arr.append([sigma, alpha, K, i, j])
    np.savez(data_dir / 'cluster_job_arr.npz',
             cluster_job_arr)
    print(len(cluster_job_arr))

def launch_multiple_individual_hmmjob(cluster_arr_sep, inpt_mod, y, session, mask, session_fold_lookup_table,
                               D, C, N_em_iters, global_fit,global_data_dir, overall_dir):
    [prior_sigma, transition_alpha, K, fold, iter] = cluster_arr_sep

    iter = int(iter)
    fold = int(fold)
    K = int(K)
    init_param_file = global_data_dir / ('best_global_params/best_params_K_' + str(K) + '.npz')

    # create save directory for this initialization/fold combination:
    save_directory = overall_dir / ('GLM_HMM_K_' + str(
        K)) / ('fold_' + str(fold)) / ('iter_' + str(iter))
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    launch_glm_hmm_job(inpt_mod, y, session, mask, session_fold_lookup_table,
                       K, D, C, N_em_iters, transition_alpha, prior_sigma,
                       fold, iter, global_fit, init_param_file,
                       save_directory)

if __name__ == '__main__':
    # if len(sys.argv)==1:
    #     print('Please specify the data folder you want')
    #     exit()
    # root_folder_dir = str(sys.argv[1])
    root_folder_dir = '/home/anh/Documents/phd'

    root_folder_name = 'om_choice_nopfail'
    root_data_dir = Path(root_folder_dir) / root_folder_name / 'data'
    root_result_dir = Path(root_folder_dir) / root_folder_name / 'result'
    global_data_dir = root_data_dir /  (root_folder_name + '_data_for_cluster')
    data_dir = global_data_dir / 'data_by_animal'


    animal_list = load_animal_list(data_dir / 'animal_list.npz')
    processed_file_name = '_processed.npz'
    session_lookup_name = '_session_fold_lookup.npz'

    # Create cluster jobs to run individual glm-hmm fit later
    create_cluster_job_individual_fit(N_initializations=2,K_vals=K_vals,num_folds=num_folds,prior_sigma=prior_sigma,transition_alpha=transition_alpha,data_dir=data_dir)

    # Create directory for results:
    results_dir = root_result_dir/ (root_folder_name + '_individual_fit')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if root_folder_name == 'om_accuracy':
        labels_for_plot = ['prev_failure', 'sound_side', 'stim','intercept']
    else:
        # labels_for_plot = ['prev-fail','stim', 'stim:prev-fail', 'prev-choice', 'bias']
        labels_for_plot = ['stim', 'stim-identity', 'prev-choice', 'bias']

    for animal in animal_list:
        # Fit GLM to data from single animal:
        animal_file = data_dir / (animal + processed_file_name)
        session_fold_lookup_table = load_session_fold_lookup(
            data_dir / (animal + session_lookup_name))

        for fold in range(num_folds):
            this_results_dir = results_dir / animal

            # Load data
            inpt, y, session = load_data(animal_file)

            figure_directory = this_results_dir / "GLM" / ("fold_" + str(fold))
            if not os.path.exists(figure_directory):
                os.makedirs(figure_directory)

            # Subset to sessions of interest for fold
            sessions_to_keep = session_fold_lookup_table[np.where(
                session_fold_lookup_table[:, 1] != fold), 0]
            idx_this_fold = [
                str(sess) in sessions_to_keep and y[id, 0] != -1
                for id, sess in enumerate(session)
            ]
            this_inpt, this_y, this_session = inpt[idx_this_fold, :], \
                                              y[idx_this_fold, :], \
                                              session[idx_this_fold]
            assert len(
                np.unique(this_y)
            ) == 2, "choice vector should only include 2 possible values"
            train_size = this_inpt.shape[0]

            M = this_inpt.shape[1]
            loglikelihood_train_vector = []

            for iter in range(N_initializations):
                loglikelihood_train, recovered_weights = fit_glm([this_inpt],
                                                                 [this_y], M,
                                                                 C)
                weights_for_plotting = append_zeros(recovered_weights)
                plot_input_vectors(weights_for_plotting,
                                   figure_directory,
                                   title="GLM fit; Final LL = " +
                                   str(loglikelihood_train),
                                   save_title='init' + str(iter),
                                   labels_for_plot=labels_for_plot)
                loglikelihood_train_vector.append(loglikelihood_train)
                np.savez(
                    figure_directory / ('variables_of_interest_iter_' +
                    str(iter) + '.npz'), loglikelihood_train, recovered_weights)
    #

    ###################################################################################
    # FIT INDIVIDUAL GLM-HMM
    # Load external files:
    cluster_arr_file = data_dir / 'cluster_job_arr.npz'
    # Load cluster array job parameters:
    cluster_arr = load_cluster_arr(cluster_arr_file)

    if multiprocess_option:
        animal_list = load_animal_list(data_dir / 'animal_list.npz')
        for i, animal in enumerate(animal_list):
            print(animal)
            animal_file = data_dir / (animal + processed_file_name)
            session_fold_lookup_table = load_session_fold_lookup(
                data_dir / (animal + session_lookup_name))

            global_fit = False

            inpt, y, session = load_data(animal_file)
            #  append a column of ones to inpt to represent the bias covariate:
            inpt_mod = np.hstack((inpt, np.ones((len(inpt), 1))))
            y = y.astype('int')

            overall_dir = results_dir / animal

            # Identify violations for exclusion:
            violation_idx = np.where(y == -1)[0]
            nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                           inpt.shape[0])

            iterables = [(cluster_arr_sep, inpt_mod, y, session, mask, session_fold_lookup_table,
                                   D, C, N_em_iters, global_fit,global_data_dir, overall_dir)
                         for cluster_arr_sep in cluster_arr]
            # initialize multiple processes
            pool = mp.Pool(n_processes)
            # launch multiple processes - starmap allows for multiple argumetns
            pool.starmap(launch_multiple_individual_hmmjob, iterables)
            # Close the pool for new tasks
            pool.close()
            # Wait for all tasks to complete at this point
            pool.join()
    else:
        # test multiprocessing fit
        # for z in range(cluster_arr.shape[0]):
        for z in range(20,30):
            [prior_sigma, transition_alpha, K, fold, iter] = cluster_arr[z]

            iter = int(iter)
            fold = int(fold)
            K = int(K)

            animal_list = load_animal_list(data_dir / 'animal_list.npz')

            for i, animal in enumerate(animal_list):
                print(animal)
                animal_file = data_dir / (animal + processed_file_name)
                session_fold_lookup_table = load_session_fold_lookup(
                    data_dir / (animal + session_lookup_name))

                global_fit = False

                inpt, y, session = load_data(animal_file)
                #  append a column of ones to inpt to represent the bias covariate:
                inpt = np.hstack((inpt, np.ones((len(inpt), 1))))
                y = y.astype('int')

                overall_dir = results_dir / animal

                # Identify violations for exclusion:
                violation_idx = np.where(y == -1)[0]
                nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                               inpt.shape[0])

                init_param_file = global_data_dir / ('best_global_params/best_params_K_' + str(K) + '.npz')

                # create save directory for this initialization/fold combination:
                save_directory = overall_dir / ('GLM_HMM_K_' + str(
                    K)) / ('fold_' + str(fold)) / ('iter_' + str(iter))
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table,
                                   K, D, C, N_em_iters, transition_alpha, prior_sigma,
                                   fold, iter, global_fit, init_param_file,
                                   save_directory)
    ###################################################################################
    # RUN INDIVIDUAL GLM-HMM POST PROCESSING

    for animal in animal_list:
        overall_dir = results_dir / animal
        # Load data
        inpt, y, session = load_data(data_dir / (animal + processed_file_name))
        session_fold_lookup_table = load_session_fold_lookup(
            data_dir / (animal + session_lookup_name))

        animal_preferred_model_dict = {}

        cvbt_folds_model = np.zeros((num_models, num_folds))
        cvbt_train_folds_model = np.zeros((num_models, num_folds))

        # Save best initialization for each model-fold combination
        best_init_cvbt_dict = {}
        for fold in range(num_folds):
            print("fold = " + str(fold))
            test_inpt, test_y, test_nonviolation_mask, \
            this_test_session, train_inpt, train_y, \
            train_nonviolation_mask, this_train_session, M, n_test, \
            n_train = prepare_data_for_cv(
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
                    # Load parameters and instantiate a new GLM
                    # object with these parameters
                    glm_weights_file = overall_dir / \
                                       'GLM' / ('fold_' + str(fold)) / 'variables_of_interest_iter_0.npz'
                    ll_glm = calculate_glm_test_loglikelihood(
                        glm_weights_file,
                        test_y[test_nonviolation_mask == 1, :],
                        test_inpt[test_nonviolation_mask == 1, :], M, C)
                    ll_glm_train = calculate_glm_test_loglikelihood(
                        glm_weights_file,
                        train_y[train_nonviolation_mask == 1, :],
                        train_inpt[train_nonviolation_mask == 1, :], M, C)
                    cvbt_folds_model[0, fold] = calculate_cv_bit_trial(
                        ll_glm, ll0, n_test)
                    cvbt_train_folds_model[0, fold] = calculate_cv_bit_trial(
                        ll_glm_train, ll0_train, n_train)
                elif model == "GLM_HMM":
                    for K in range(2, K_max+1):
                        print("K = "+ str(K))
                        model_idx = K-1
                        cvbt_folds_model[model_idx, fold], \
                        cvbt_train_folds_model[
                            model_idx, fold], _, _, \
                        init_ordering_by_train = return_glmhmm_nll(
                            np.hstack((inpt, np.ones((len(inpt), 1)))), y, session,
                            session_fold_lookup_table, fold, K, D, C,
                            overall_dir)
                        # Save best initialization to dictionary for
                        # later:
                        key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(
                            fold)
                        best_init_cvbt_dict[key_for_dict] = int(
                            init_ordering_by_train[0])
        # Save best initialization directories across animals,
        # folds and models (only GLM-HMM):
        print(cvbt_folds_model)
        print(cvbt_train_folds_model)
        json_dump = json.dumps(best_init_cvbt_dict)
        f = open(overall_dir / "best_init_cvbt_dict.json", "w")
        f.write(json_dump)
        f.close()
        # Save cvbt_folds_model as numpy array for easy parsing
        # across all models and folds
        np.savez(overall_dir / "cvbt_folds_model.npz", cvbt_folds_model)
        np.savez(overall_dir / "cvbt_train_folds_model.npz",
                 cvbt_train_folds_model)