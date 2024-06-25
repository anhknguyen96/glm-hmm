#  Fit GLM to all IBL data together

import autograd.numpy as np
import autograd.numpy.random as npr
import os
import multiprocessing as mp
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

def launch_multiple_hmmjob(cluster_arr_sep, results_dir, inpt_mod, y, session,mask,
                       session_fold_lookup_table,D,C,N_em_iters,transition_alpha,prior_sigma,global_fit):

    K, fold, iter = cluster_arr_sep
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

    # K=3, fold = 2, iter = 0
    cluster_test = cluster_arr[140:]

    iterables = [(cluster_arr_sep, results_dir, inpt_mod, y, session,mask,
                       session_fold_lookup_table,D,C,N_em_iters,transition_alpha,prior_sigma,global_fit)
                 for cluster_arr_sep in cluster_test]
    # iterables = zip(*iterables)


    n_processes = 15
    # initialize multiple processes
    pool = mp.Pool(n_processes)
    # launch multiple processes
    pool.starmap(launch_multiple_hmmjob, iterables)
    # Close the pool for new tasks
    pool.close()
    # Wait for all tasks to complete at this point
    pool.join()
