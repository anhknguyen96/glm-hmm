import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from preprocessing_utils import create_train_test_sessions

root_folder_name = 'om_accuracy'
root_data_dir = Path('../../data')
data_dir = root_data_dir / root_folder_name / (root_folder_name +'_data_for_cluster')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
# Also create a subdirectory for storing each individual animal's data:
if not os.path.exists(data_dir / "data_by_animal"):
    os.makedirs(data_dir / "data_by_animal")
if not os.path.exists(data_dir / "partially_processed"):
    os.makedirs(data_dir / "partially_processed")

file_path = '/home/anh/Documents/phd/outcome_manip_git/data/om_all_batch1&2&3_phase4.csv'
om_cleaned = pd.read_csv(file_path)
om_cleaned['mouse_id'] = om_cleaned['mouse_id'].astype(str)
om_cleaned['session_identifier'] =  'm' + om_cleaned['mouse_id']+ 's' + om_cleaned['session_no'].astype(str)
animal_df = om_cleaned[['mouse_id','session_identifier']].copy()
animal_df = animal_df.drop_duplicates(subset=['session_identifier'])
animal_eid_dict = animal_df.groupby('mouse_id')['session_identifier'].apply(list).to_dict()
animal_list = om_cleaned.mouse_id.unique()
json = json.dumps(animal_eid_dict)
f = open(data_dir / 'partially_processed' /'animal_eid_dict.json',  "w")
f.write(json)
f.close()
np.savez(data_dir / 'partially_processed'/ 'animal_list.npz', animal_list)
choice_or_accuracy = 'acc'
if choice_or_accuracy == 'acc':
    columns_wanted = ['prev_failure', 'sound_index','z_abs_freqs','success']
else:
    columns_wanted = ['freq_trans','prev_choice','wlsw','lick_side_freq']
for mouse_id in range(len(animal_list)):
    om_tmp = om_cleaned.loc[om_cleaned['mouse_id'] == animal_list[mouse_id]].copy().reset_index()
    T = len(om_tmp)
    design_mat = np.zeros((T, len(columns_wanted)-1))
    for design_mat_arr_index in range(len(columns_wanted)):
        if design_mat_arr_index != len(columns_wanted)-1:
            design_mat[:, design_mat_arr_index] = np.array(om_tmp[columns_wanted[design_mat_arr_index]])
        else:
            y = np.expand_dims(np.array(om_tmp[columns_wanted[design_mat_arr_index]]), axis=1)
            y = y.astype('int')         # assertion error in ssm stats
    session = np.array(om_tmp.session_identifier)
    rewarded = np.array(om_tmp.success)
    np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_id] + choice_or_accuracy + '_processed.npz'),
             design_mat, y,
             session)
    np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_id] + '_rewarded.npz'),
             rewarded)
    animal_session_fold_lookup = create_train_test_sessions(session,
                                                            5)
    if mouse_id == 0:
        master_session_fold_lookup_table = animal_session_fold_lookup
        master_y = np.copy(y)
        master_inpt = design_mat
        master_session = session
        master_rewarded = rewarded
    else:
        master_y = np.vstack((master_y, y))
        master_inpt = np.vstack((master_inpt, design_mat))
        master_session= np.concatenate((master_session,session))
        master_rewarded = np.concatenate((master_rewarded,rewarded))
        master_session_fold_lookup_table = np.vstack(
        (master_session_fold_lookup_table, animal_session_fold_lookup))
    np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_id] + choice_or_accuracy +
             "_session_fold_lookup" +
             ".npz"),
             animal_session_fold_lookup)

np.savez(data_dir / (choice_or_accuracy+'_all_animals_concat.npz'),
             master_inpt,
             master_y, master_session)
np.savez(data_dir / 'all_animals_concat_rewarded.npz',
             master_rewarded)
np.savez(
    data_dir / 'all_animals_concat_session_fold_lookup.npz',
    master_session_fold_lookup_table)
#
# # this is for when the animal minmum data condition is considered
# np.savez(data_dir / 'data_by_animal' / 'animal_list.npz',
#          animal_list)
# final_animal_eid_dict = animal_eid_dict
# json = json.dumps(final_animal_eid_dict)
# f = open(data_dir / "final_animal_eid_dict.json", "w")
# f.write(json)
# f.close()