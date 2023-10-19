import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from patsy import dmatrices
from preprocessing_utils import create_train_test_sessions

file_path = '/home/anh/Documents/phd/outcome_manip_git/data/om_all_batch1&2&3&4.csv'

# this has to be changed if want to run the model on different datasets
root_folder_name = 'om_choice'
root_data_dir = Path('../../data')
data_dir = root_data_dir / root_folder_name / (root_folder_name +'_data_for_cluster')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Also create a subdirectory for storing each individual animal's data:
if not os.path.exists(data_dir / "data_by_animal"):
    os.makedirs(data_dir / "data_by_animal")
if not os.path.exists(data_dir / "partially_processed"):
    os.makedirs(data_dir / "partially_processed")

# read file and preprocess data for glm-hmm
om = pd.read_csv(file_path)
# only get engaged data
om_cleaned = om.loc[(om['lick_side_freq']!=-2)&(om['prev_lick_side_freq']!=-2)].reset_index()
om_cleaned['prev_failure'] = om_cleaned['prev_failure'].astype('int')
om_cleaned['mouse_id'] = om_cleaned['mouse_id'].astype(str)
# om_cleaned['session_identifier'] =  'm' + om_cleaned['mouse_id']+ 's' + om_cleaned['session_no'].astype(str)

# to create a dict of mice
animal_df = om_cleaned[['mouse_id','session_identifier']].copy()
animal_df = animal_df.drop_duplicates(subset=['session_identifier'])
animal_eid_dict = animal_df.groupby('mouse_id')['session_identifier'].apply(list).to_dict()
animal_list = om_cleaned.mouse_id.unique()
json = json.dumps(animal_eid_dict)
f = open(data_dir / 'partially_processed' /'animal_eid_dict.json',  "w")
f.write(json)
f.close()
np.savez(data_dir / 'partially_processed'/ 'animal_list.npz', animal_list)

# create predictors matrix for model fitting
choice_or_accuracy = 'choice'
if choice_or_accuracy == 'acc':
    formula = 'success ~ -1 + standardize(abs_freq) + C(sound_index) + C(prev_failure)'
    formula_unnormalized = 'success ~ -1 + abs_freq + C(sound_index) + C(prev_failure)'
else:
    formula = 'lick_side_freq ~ -1 + standardize(freq_trans) + C(prev_failure) + standardize(freq_trans):C(prev_failure) + C(prev_lick_side_freq)'
    formula_unnormalized = 'lick_side_freq ~ -1 + freq_trans + C(prev_failure) + freq_trans:C(prev_failure) + C(prev_lick_side_freq)'
for mouse_id in range(len(animal_list)):
    # subselect and clean data based on mouse id
    om_tmp = om_cleaned.loc[om_cleaned['mouse_id'] == animal_list[mouse_id]].copy().reset_index()
    T = len(om_tmp)
    # create predictor matrix from formula using patsy
    outcome, predictors = dmatrices(formula_unnormalized, om_tmp, return_type='dataframe')
    get_col = predictors.columns[1:].to_list()
    design_mat = np.asarray(predictors[get_col])

    _, predictors_unnorm = dmatrices(formula, om_tmp, return_type='dataframe')
    get_col_unnorm = predictors_unnorm.columns[1:].to_list()
    design_mat_unnorm = np.asarray(predictors_unnorm[get_col_unnorm])

    y = np.asarray(outcome).astype('int')         # assertion error in ssm stats
    session = np.array(om_tmp.session_identifier)
    rewarded = np.array(om_tmp.success)

    # np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_id] + choice_or_accuracy + '_processed.npz'),
    #          design_mat, y,
    #          session)
    np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_id] + choice_or_accuracy + '_unnormalized.npz'),
             design_mat_unnorm, y,
             session)
    # np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_id] + '_rewarded.npz'),
    #          rewarded)
    # animal_session_fold_lookup = create_train_test_sessions(session,
    #                                                         5)
    # if mouse_id == 0:
    #     master_session_fold_lookup_table = animal_session_fold_lookup
    #     master_y = np.copy(y)
    #     master_inpt = design_mat
    #     master_session = session
    #     master_rewarded = rewarded
    # else:
    #     master_y = np.vstack((master_y, y))
    #     master_inpt = np.vstack((master_inpt, design_mat))
    #     master_session= np.concatenate((master_session,session))
    #     master_rewarded = np.concatenate((master_rewarded,rewarded))
    #     master_session_fold_lookup_table = np.vstack(
    #     (master_session_fold_lookup_table, animal_session_fold_lookup))
    # np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_id] + choice_or_accuracy +
    #          "_session_fold_lookup" +
    #          ".npz"),
    #          animal_session_fold_lookup)
print(predictors_unnorm.columns)
# np.savez(data_dir / (choice_or_accuracy+'_all_animals_concat.npz'),
#              master_inpt,
#              master_y, master_session)
# np.savez(data_dir / 'all_animals_concat_rewarded.npz',
#              master_rewarded)
# np.savez(
#     data_dir / 'all_animals_concat_session_fold_lookup.npz',
#     master_session_fold_lookup_table)
#
# # this is for when the animal minmum data condition is considered
# np.savez(data_dir / 'data_by_animal' / 'animal_list.npz',
#          animal_list)
# final_animal_eid_dict = animal_eid_dict
# json = json.dumps(final_animal_eid_dict)
# f = open(data_dir / "final_animal_eid_dict.json", "w")
# f.write(json)
# f.close()