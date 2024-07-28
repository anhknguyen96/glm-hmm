# Calculate predictive accuracy for each individual animal (for Figure 2c
# and 4b).  Note: this is the same code for both figures, and only needs to
# be run once to calculate the quantities required to generate both figures
import json
import sys
from pathlib import Path
import numpy as np
import numpy.random as npr

from plotting_utils import load_glmhmm_data, load_animal_list, load_cv_arr, \
    load_data, load_glm_vectors, load_lapse_params, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, create_train_test_trials_for_pred_acc, \
    calculate_predictive_accuracy, calculate_predictive_acc_glm, \
    calculate_predictive_acc_lapse_model

cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306"]

if __name__ == '__main__':
    root_folder_dir = '/home/anh/Documents'
    root_folder_name = 'om_choice'
    root_data_dir = Path(root_folder_dir) / root_folder_name / 'data'
    root_result_dir = Path(root_folder_dir) / root_folder_name / 'result'

    data_dir = root_data_dir / (root_folder_name + '_data_for_cluster')
    data_individual = data_dir / 'data_by_animal'
    results_dir = root_result_dir / (root_folder_name + '_global_fit')
    results_dir_individual = root_result_dir / (root_folder_name + '_individual_fit')
    animal_list = load_animal_list(data_individual / 'animal_list.npz')

    sigma_val = 2
    alpha_val = 2
    K_max = 5
    npr.seed(67)

    for animal in animal_list:
        print(animal)
        results_dir_individual_animal = results_dir_individual / animal

        cv_file = results_dir_individual_animal / "cvbt_folds_model.npz"
        cvbt_folds_model = load_cv_arr(cv_file)

        predictive_acc_mat = []
        num_trials_mat = []

        # Also get data for animal:
        inpt, y, session = load_data(data_individual / (animal + '_processed.npz'))

        # create train test idx
        trial_fold_lookup_table = create_train_test_trials_for_pred_acc(
            y, num_folds=5)

        # GLM fit:
        # Load params:
        _, glm_weights = load_glm_vectors(
            results_dir_individual_animal / 'GLM' / 'fold_0' / 'variables_of_interest_iter_0.npz')
        predictive_acc_glm = []
        for fold in range(5):
            # identify the idx for exclusion:
            idx_to_exclude = trial_fold_lookup_table[np.where(
                trial_fold_lookup_table[:, 1] == fold)[0], 0].astype('int')
            predictive_acc = calculate_predictive_acc_glm(
                glm_weights, inpt, y, idx_to_exclude)
            predictive_acc_glm.append(predictive_acc)
            num_trials_mat.append(len(idx_to_exclude))
        predictive_acc_mat.append(predictive_acc_glm)

        # GLM-HMM fit:
        for K in range(2, K_max+1):
            with open(results_dir_individual_animal / "best_init_cvbt_dict.json", 'r') as f:
                best_init_cvbt_dict = json.load(f)

                # Get the file name corresponding to the best initialization
                # for given K value
                raw_file = get_file_name_for_best_model_fold(
                    cvbt_folds_model, K, results_dir_individual_animal, best_init_cvbt_dict)
                hmm_params, lls = load_glmhmm_data(raw_file)

                # Save parameters for initializing individual fits
                weight_vectors = hmm_params[2]
                log_transition_matrix = hmm_params[1][0]
                init_state_dist = hmm_params[0][0]

                predictive_acc_this_K = []
                for fold in range(5):
                    # identify the idx for exclusion:
                    idx_to_exclude = trial_fold_lookup_table[np.where(
                        trial_fold_lookup_table[:, 1] == fold)[0],
                                                             0].astype('int')
                    # Make a copy of y and modify idx
                    y_modified = np.copy(y)
                    y_modified[idx_to_exclude] = -1
                    # Create mask:
                    # Identify violations for exclusion:
                    violation_idx = np.where(y_modified == -1)[0]
                    nonviolation_idx, mask = create_violation_mask(
                        violation_idx, inpt.shape[0])
                    y_modified[np.where(y_modified == -1), :] = 1
                    inputs, datas, train_masks = partition_data_by_session(
                        np.hstack((inpt, np.ones((len(inpt), 1)))), y_modified,
                        mask, session)
                    predictive_acc = calculate_predictive_accuracy(
                        inputs, datas, train_masks, hmm_params, K, range(K),
                        alpha_val, sigma_val, y, idx_to_exclude)
                    predictive_acc_this_K.append(predictive_acc)
                predictive_acc_mat.append(predictive_acc_this_K)
        predictive_acc_mat = np.array(predictive_acc_mat)
        np.savez(results_dir_individual_animal / "predictive_accuracy_mat.npz",
                 np.array(predictive_acc_mat))
        np.savez(results_dir_individual_animal / "correct_incorrect_mat.npz",
                 np.array(predictive_acc_mat * num_trials_mat), num_trials_mat)
