# Fit GLM to each IBL animal separately
import autograd.numpy as np
import autograd.numpy.random as npr
import os
from pathlib import Path
from glm_utils import load_session_fold_lookup, load_data, load_animal_list, \
    fit_glm, plot_input_vectors, append_zeros

npr.seed(65)

C = 2  # number of output types/categories
N_initializations = 10

if __name__ == '__main__':
    # if len(sys.argv)==1:
    #     print('Please specify the data folder you want')
    #     exit()
    # root_folder_name = str(sys.argv[1])

    root_folder_name = 'om_choice'
    root_data_dir = Path('../../data')
    root_result_dir = Path('../../results')
    data_dir = root_data_dir / root_folder_name / (root_folder_name +'_data_for_cluster') / 'data_by_animal'

    num_folds = 5
    animal_list = load_animal_list(data_dir / 'animal_list.npz')

    results_dir = root_result_dir / root_folder_name / (root_folder_name + '_individual_fit')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if root_folder_name == 'om_accuracy':
        labels_for_plot = ['prev_failure', 'sound_side', 'stim','intercept']
        processed_file_name = 'acc_processed.npz'
        session_lookup_name = 'acc_session_fold_lookup.npz'
    else:
        labels_for_plot = ['prev-fail', 'prev-choice', 'stim', 'stim:prev-fail','bias']
        processed_file_name = 'choice_processed.npz'
        session_lookup_name = 'choice_session_fold_lookup.npz'

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
