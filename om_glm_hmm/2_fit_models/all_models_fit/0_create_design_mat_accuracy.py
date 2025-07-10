import pandas as pd
import json
import os
import sys
from pathlib import Path
import numpy as np
from patsy import dmatrices
from preprocessing_utils import create_train_test_sessions
import scipy

file_path = '/home/anh/Documents/phd/outcome_manip_git/data/om_all_batch1&2&3&4_rawrows.csv'
if __name__ == '__main__':
    # if len(sys.argv)==1:
    #     print('Please specify the data folder you want')
    #     exit()
    # root_folder_dir = str(sys.argv[1])
    root_folder_dir = '/home/anh/Documents/phd'
    # root_folder_name = 'om_choice'
    pfail = 0
    root_folder_name = 'om_choice_nopfail'              # another glm0h
    root_data_dir = Path(root_folder_dir) / root_folder_name / 'data'
    root_result_dir = Path(root_folder_dir) / root_folder_name / 'result'
    data_dir = root_data_dir / (root_folder_name + '_data_for_cluster')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Also create a subdirectory for storing each individual animal's data:
    if not os.path.exists(data_dir / "data_by_animal"):
        os.makedirs(data_dir / "data_by_animal")
        os.makedirs(data_dir / "partially_processed")

    # read file and preprocess data for glm-hmm
    om = pd.read_csv(file_path)
    # clean data
    data_batch12 = om.loc[(om.mouse_id < 13) & (om.lick_side_freq != -2) & (om.prev_choice != -2) & (om.prev_reward_prob == 0.5) & (om.prev_choice2 != -2)]
    data_batch34 = om.loc[(om.mouse_id > 13) & (om.prev_om_gen == 0) & (om.lick_side_freq != -2) & (om.prev_choice != -2) & (om.prev_reward_prob == 0.5) & (om.prev_choice2 != -2)]
    om_cleaned = pd.concat([data_batch12,data_batch34],ignore_index=True)
    # now take care of predictors
    index = om_cleaned.index
    om_cleaned['prev_failure'] = om_cleaned['prev_failure'].astype('int')
    om_cleaned['mouse_id'] = om_cleaned['mouse_id'].astype(str)
    # for glmhmm, better to not z-score on a session basis -> biased estimation of intercept
    om_cleaned['glmhmm_freq_trans'] = scipy.stats.zscore(om_cleaned['freq_trans'])
    om_cleaned['z_freq_trans'] = om_cleaned['freq_trans'].copy()
    om_cleaned['z_prev_choice'] = om_cleaned['prev_choice'].copy()
    om_cleaned['z_prev_failure'] = om_cleaned['prev_failure'].copy()
    # also discard sess that have less than 50 trials
    om_cleaned_session = pd.DataFrame()
    for session_no in om_cleaned.session_identifier.unique():
        # get indices of trials in the session
        session_no_index = list(index[(om_cleaned['session_identifier'] == session_no)])
        if len(session_no_index) <= 50:
            print(session_no)
            continue
        # z score predictors on a session basis
        om_cleaned.loc[session_no_index, 'z_freq_trans'] = scipy.stats.zscore(
            om_cleaned.loc[session_no_index, 'freq_trans'])
        median_sess = om_cleaned.loc[session_no_index, 'z_freq_trans'].median()
        # inplace does not work with df slices!!
        om_cleaned.loc[session_no_index, 'z_freq_trans'] = om_cleaned.loc[
            session_no_index, 'z_freq_trans'].fillna(median_sess)

        om_cleaned.loc[session_no_index, 'z_prev_choice'] = scipy.stats.zscore(
            om_cleaned.loc[session_no_index, 'prev_choice'])
        median_sess = om_cleaned.loc[session_no_index, 'z_prev_choice'].median()
        # inplace does not work with df slices!!
        om_cleaned.loc[session_no_index, 'z_prev_choice'] = om_cleaned.loc[
            session_no_index, 'z_prev_choice'].fillna(median_sess)

        om_cleaned.loc[session_no_index, 'z_prev_failure'] = scipy.stats.zscore(
            om_cleaned.loc[session_no_index, 'prev_failure'])
        #has to be mean for prev failure because median spits out nan
        median_sess = om_cleaned.loc[session_no_index, 'z_prev_failure'].mean()
        # inplace does not work with df slices!!
        om_cleaned.loc[session_no_index, 'z_prev_failure'] = om_cleaned.loc[
            session_no_index, 'z_prev_failure'].fillna(median_sess)

        # only get sesssions that have more than 50 trials
        om_cleaned_session = pd.concat((om_cleaned_session,om_cleaned.loc[om_cleaned.session_identifier == session_no]))
    # save for other processes
    om_cleaned_session = om_cleaned_session.reset_index()
    print(len(om_cleaned))
    print(len(om_cleaned_session))
    om_cleaned_session.to_csv(os.path.join(data_dir,'om_all_batch1&2&3&4_processed.csv'))
    del om_cleaned, om, data_batch34, data_batch12

    # to create a dict of mice
    animal_df = om_cleaned_session[['mouse_id','session_identifier']].copy()
    animal_df = animal_df.drop_duplicates(subset=['session_identifier'])
    animal_eid_dict = animal_df.groupby('mouse_id')['session_identifier'].apply(list).to_dict()
    animal_list = om_cleaned_session.mouse_id.unique()
    json = json.dumps(animal_eid_dict)
    f = open(data_dir / 'partially_processed' /'animal_eid_dict.json',  "w"); f.write(json); f.close()
    f = open(data_dir / 'data_by_animal' / 'animal_eid_dict.json', "w"); f.write(json); f.close()
    np.savez(data_dir / 'partially_processed'/ 'animal_list.npz', animal_list)
    np.savez(data_dir / 'data_by_animal' / 'animal_list.npz', animal_list)

    # create predictors matrix for model fitting
    choice_or_accuracy = 'choice'
    if choice_or_accuracy == 'acc':
        formula = 'success ~ -1 + z_freq + C(prev_failure)'
        formula_unnormalized = 'success ~ -1 + abs_freq + C(prev_failure)'
    else:
        if pfail:
            formula = 'lick_side_freq ~ -1 + glmhmm_freq_trans + C(prev_failure) + z_freq_trans:C(prev_failure) + prev_choice'
            formula_unnormalized = 'lick_side_freq ~ -1 + freq_trans + C(prev_failure) + freq_trans:C(prev_failure) + prev_choice'
        else:
            formula = 'lick_side_freq ~ -1 + glmhmm_freq_trans + prev_choice'
            formula_unnormalized = 'lick_side_freq ~ -1 + freq_trans + prev_choice'
    for mouse_index in range(len(animal_list)):
        # subselect and clean data based on mouse id
        om_tmp = om_cleaned_session.loc[om_cleaned_session['mouse_id'] == animal_list[mouse_index]].copy().reset_index()
        T = len(om_tmp)
        # create predictor matrix from formula using patsy
        outcome, predictors = dmatrices(formula, om_tmp, return_type='dataframe')
        print(predictors.columns)
        # skip the first column because it is pfail 0
        get_col = predictors.columns[pfail:].to_list()
        design_mat = np.asarray(predictors[get_col])
        # create predictor matrix from formula using patsy - unnormalized predictors
        _, predictors_unnorm = dmatrices(formula_unnormalized, om_tmp, return_type='dataframe')
        print(predictors_unnorm.columns)
        # skip the first column because it is pfail 0
        get_col_unnorm = predictors_unnorm.columns[pfail:].to_list()
        design_mat_unnorm = np.asarray(predictors_unnorm[get_col_unnorm])

        y = np.asarray(outcome).astype('int')         # assertion error in ssm stats
        session = np.array(om_tmp.session_identifier)
        rewarded = np.array(om_tmp.success)

        np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_index] + '_processed.npz'),
                 design_mat, y,
                 session)
        np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_index] + '_unnormalized.npz'),
                 design_mat_unnorm, y,
                 session)
        np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_index] + '_rewarded.npz'),
                 rewarded)
        animal_session_fold_lookup = create_train_test_sessions(session,
                                                                5)
        if mouse_index == 0:
            master_session_fold_lookup_table = animal_session_fold_lookup
            master_y = np.copy(y)
            master_inpt = design_mat
            master_inpt_unnorm = design_mat_unnorm
            master_session = session
            master_rewarded = rewarded
        else:
            master_y = np.vstack((master_y, y))
            master_inpt = np.vstack((master_inpt, design_mat))
            master_inpt_unnorm = np.vstack((master_inpt_unnorm, design_mat_unnorm))
            master_session= np.concatenate((master_session,session))
            master_rewarded = np.concatenate((master_rewarded,rewarded))
            master_session_fold_lookup_table = np.vstack(
            (master_session_fold_lookup_table, animal_session_fold_lookup))
        np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_index] +
                 "_session_fold_lookup" +
                 ".npz"),
                 animal_session_fold_lookup)
    print(predictors_unnorm.columns)
    np.savez(data_dir / 'all_animals_concat.npz',
                 master_inpt,
                 master_y, master_session)
    np.savez(data_dir / 'all_animals_concat_unnormalized.npz',
             master_inpt,
             master_y, master_session)
    np.savez(data_dir / 'all_animals_concat_rewarded.npz',
                 master_rewarded)
    np.savez(
        data_dir / 'all_animals_concat_session_fold_lookup.npz',
        master_session_fold_lookup_table)