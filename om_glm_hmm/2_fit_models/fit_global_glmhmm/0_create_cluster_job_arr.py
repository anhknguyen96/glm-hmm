
#  In order to facilitate parallelization of jobs, create a job array that
#  can be used on e.g. a cluster
import numpy as np
from pathlib import Path
K_vals = [2, 3, 4, 5]
num_folds = 5
N_initializations = 20
root_folder_name = 'om_accuracy'
root_data_dir = Path('../../data')
root_result_dir = Path('../../results')

data_dir = root_data_dir / root_folder_name / (root_folder_name + '_data_for_cluster')

if __name__ == '__main__':
    cluster_job_arr = []
    for K in K_vals:
        for i in range(num_folds):
            for j in range(N_initializations):
                cluster_job_arr.append([K, i, j])
    np.savez(data_dir / "cluster_job_arr.npz",
             cluster_job_arr)
