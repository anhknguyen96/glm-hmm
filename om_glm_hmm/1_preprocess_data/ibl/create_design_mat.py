import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from preprocessing_utils import create_train_test_sessions

root_folder_name = 'om_choice_batch3'
root_data_dir = Path('../../data')
data_dir = root_data_dir / root_folder_name / (root_folder_name +'_data_for_cluster')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
# Also create a subdirectory for storing each individual animal's data:
if not os.path.exists(data_dir / "data_by_animal"):
    os.makedirs(data_dir / "data_by_animal")
if not os.path.exists(data_dir / "partially_processed"):
    os.makedirs(data_dir / "partially_processed")

file_path = '/home/anh/Documents/phd/outcome_manip/data/om_batch_3_choice_glmhmm.csv'
om_cleaned = pd.read_csv(file_path)
om_cleaned['mouse_id'] = om_cleaned['mouse_id'].astype(str)
animal_df = om_cleaned[['mouse_id','session_identifier']].copy()
animal_df = animal_df.drop_duplicates(subset=['session_identifier'])
animal_eid_dict = animal_df.groupby('mouse_id')['session_identifier'].apply(list).to_dict()
animal_list = om_cleaned.mouse_id.unique()
json = json.dumps(animal_eid_dict)
f = open(data_dir / 'partially_processed' /'animal_eid_dict.json',  "w")
f.write(json)
f.close()
np.savez(data_dir / 'partially_processed'/ 'animal_list.npz', animal_list)

# # Require that each animal has at least 30 sessions (=2700 trials) of data:
# req_num_sessions = 30  # 30*90 = 2700
# for animal in animal_list:
#     num_sessions = len(animal_eid_dict[animal])
#     if num_sessions < req_num_sessions:
#         animal_list = np.delete(animal_list,


for mouse_id in range(len(animal_list)):
    om_tmp = om_cleaned.loc[om_cleaned['mouse_id'] == animal_list[mouse_id]].copy().reset_index()
    T = len(om_tmp)
    design_mat = np.zeros((T, 3))
    design_mat[:, 0] = np.array(om_tmp.z_freq)
    design_mat[:, 1] = np.array(om_tmp.prev_choice)
    design_mat[:, 2] = np.array(om_tmp.wslw)
    y = np.expand_dims(np.array(om_tmp.lick_side_freq), axis=1)
    session = np.array(om_tmp.session_identifier)
    np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_id] + '_processed.npz'),
             design_mat, y,
             session)
    animal_session_fold_lookup = create_train_test_sessions(session,
                                                            5)
    if mouse_id == 0:
        master_session_fold_lookup_table = animal_session_fold_lookup
    else:
        master_session_fold_lookup_table = np.vstack(
        (master_session_fold_lookup_table, animal_session_fold_lookup))
    np.savez(data_dir / 'data_by_animal' / (animal_list[mouse_id] +
             "_session_fold_lookup" +
             ".npz"),
             animal_session_fold_lookup)

T_all = len(om_cleaned)
design_mat = np.zeros((T_all, 3))
design_mat[:, 0] = np.array(om_cleaned.z_freq)
design_mat[:, 1] = np.array(om_cleaned.prev_choice)
design_mat[:, 2] = np.array(om_cleaned.wslw)
y = np.expand_dims(np.array(om_cleaned.lick_side_freq), axis=1)
session = np.array(om_cleaned.session_identifier)
np.savez(data_dir / 'all_animals_concat.npz',
             design_mat,
             y, session)
np.savez(
    data_dir / 'all_animals_concat_session_fold_lookup.npz',
    master_session_fold_lookup_table)

# this is for when the animal minmum data condition is considered
np.savez(data_dir / 'data_by_animal' / 'animal_list.npz',
         animal_list)
# final_animal_eid_dict = animal_eid_dict
# json = json.dumps(final_animal_eid_dict)
# f = open(data_dir / "final_animal_eid_dict.json", "w")
# f.write(json)
# f.close()