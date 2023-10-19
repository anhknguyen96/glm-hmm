# Create Figure 3d - fractional occupancies of 3 states
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')

from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct

if __name__ == '__main__':

    root_folder_name = 'om_choice'
    root_data_dir = Path('../../data')
    root_result_dir = Path('../../results')
    root_figure_dir = Path('../../figures')

    animal = "12.0"
    K = 2

    global_data_dir = root_data_dir / root_folder_name / (root_folder_name + '_data_for_cluster')
    data_dir = global_data_dir / 'data_by_animal'
    overall_dir = root_result_dir / root_folder_name / (root_folder_name + '_individual_fit')
    results_dir = overall_dir / animal
    figure_dir = root_figure_dir / root_folder_name / 'figure_3'
    if not Path.exists(figure_dir):
        Path.mkdir(figure_dir, parents=True)
    # data_dir = '../../data/ibl/data_for_cluster/data_by_animal/'
    # results_dir = '../../results/ibl_individual_fit/' + animal + '/'
    # figure_dir = '../../figures/figure_3/'
    if root_folder_name == 'om_accuracy':
        processed_file_name = 'acc_processed.npz'
    else:
        processed_file_name = 'choice_processed.npz'


    inpt, y, session = load_data(data_dir /(animal + processed_file_name))
    # unnormalized_inpt, _, _ = load_data(data_dir + animal +
    #                                     '_unnormalized.npz')

    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                   inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, masks = partition_data_by_session(
        np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

    # get params for posterior probs
    cv_file = results_dir / "cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(results_dir / "best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K
    # value
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                 results_dir,
                                                 best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)

    posterior_probs = get_marginal_posterior(inputs, datas, masks,
                                             hmm_params, K, range(K))

    states_max_posterior = np.argmax(posterior_probs, axis=1)
    state_occupancies = []
    cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306",
            '#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00'
            ]
    for k in range(K):
        # Get state occupancy:
        occ = len(
            np.where(states_max_posterior == k)[0]) / len(states_max_posterior)
        state_occupancies.append(occ)

    # ====================== PLOTTING CODE ===============================
    plt_xticks_location = np.arange(K)
    plt_xticks_label = [str(x) for x in (plt_xticks_location + 1)]
    fig = plt.figure(figsize=(1.3, 1.7))
    plt.subplots_adjust(left=0.4, bottom=0.3, right=0.95, top=0.95)
    for z, occ in enumerate(state_occupancies):
        plt.bar(z, occ, width=0.8, color=cols[z])
    plt.ylim((0, 1))
    plt.xticks(plt_xticks_location, plt_xticks_label, fontsize=10)
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
    plt.xlabel('state', fontsize=10)
    plt.ylabel('frac. occupancy', fontsize=10)  #, labelpad=0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(figure_dir /('fig3d_'+root_folder_name+'_'+animal+'K'+str(K)+'.png'), format='png', bbox_inches="tight")
