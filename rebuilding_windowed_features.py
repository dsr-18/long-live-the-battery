import numpy as np
import pandas as pd
import pickle
from os.path import join
import re
import math

from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

import trainer.constants as cst


WINDOW_SIZE = 50
W_SHIFT = 5
W_STRIDE = 1
# shift always at 5, stride:  1 for 10/20, 2 for 50, 4 for 100

def build_windowed_feature_df(preprocessed_pkl, window_size, shift, stride, debug=False):    
    """Returns a pandas DataFrame with all originally used features out of a loaded batch dict, 
       organized by windows"""
    print("window_size={}, shift={}, stride={}".format(window_size, shift, stride))
    cell_dfs = []

    for cell_k, cell_v in preprocessed_pkl.items():
        print('processing', cell_k)

        cell_k_pattern = re.split('c', cell_k)
        cell_batch = int(cell_k_pattern[0].split('b')[1])
        cell_num = int(cell_k_pattern[1])

        cell_cycles = cell_v['cycles']
        cell_summary = cell_v['summary']

        # slice cycle keys into windows
        cycle_keys = list(cell_cycles.keys())
        window_cycle_keys = []
        for i, w_slice in enumerate(range(0, len(cycle_keys), shift)):
            cycle_keys_slice = cycle_keys[w_slice : (w_slice + window_size) : stride]
            slice_size = len(cycle_keys_slice)
            if slice_size % (window_size//stride) != 0:     # drop remainder
                print(w_slice, 'skip short slice of size', slice_size)
                break
            else:
                window_cycle_keys.append(cycle_keys_slice)
        
        # init value arrays
        total_cycles = len(cell_cycles)
        num_windows = len(window_cycle_keys)
        assert_n = int((total_cycles-window_size) // shift) + 1
        assert math.isclose(num_windows, assert_n, rel_tol=1), "num_windows should be {}, but was {}".format(assert_n, num_windows)
        print('num_windows', num_windows, 'total_cycles', total_cycles)
        minimum_dQ_window = np.zeros(num_windows)
        variance_dQ_window = np.zeros(num_windows)
        skewness_dQ_window = np.zeros(num_windows)
        kurtosis_dQ_window = np.zeros(num_windows)
        slope_lin_fit_window = np.zeros(num_windows)
        intercept_lin_fit_window = np.zeros(num_windows)
        discharge_capacity_1 = np.zeros(num_windows)
        diff_discharge_capacity_max_1 = np.zeros(num_windows)
        mean_discharge_time = np.zeros(num_windows)
        minimum_IR_window = np.zeros(num_windows)
        diff_IR_window = np.zeros(num_windows)
        target_remaining = np.zeros(num_windows)
        target_current = np.zeros(num_windows)
        target_classifier = np.zeros(num_windows)
    
        # build cell-level df
        for i, window_keys in enumerate(window_cycle_keys):
            key_c1 = window_keys[0]
            key_clast = window_keys[-1]
            # summary keys may not line up with cycle keys, since cycles could be cleaned up
            summary_key_c1 = np.where(np.array(cycle_keys)==key_c1)[0][0]
            summary_key_clast = np.where(np.array(cycle_keys)==key_clast)[0][0]
            if debug:
                print("{}: [{}, {}], summary_keys: [{}, {}]".format(i, key_c1, key_clast, summary_key_c1, summary_key_clast))

            # 1. delta_Q_100_10(V) -> delta_Q_window(V)
            dQ_window = cell_cycles[key_clast]['Qdlin'] - cell_cycles[key_c1]['Qdlin']
            minimum_dQ_window[i] = np.log(np.abs(np.min(dQ_window)))
            variance_dQ_window[i] = np.log(np.var(dQ_window))
            skewness_dQ_window[i] = np.log(np.abs(skew(dQ_window)))
            kurtosis_dQ_window[i] = np.log(np.abs(kurtosis(dQ_window)))

            # 2. Discharge capacity fade curve features
            # Compute linear fit for cycles 2 to last:
            # discharge cappacities; q.shape = (window_size, 1); 
            q = cell_summary['QD'][summary_key_c1:summary_key_clast+1].reshape(-1, 1).astype(np.float64) 
            # Cylce index from 2 to last; X.shape = (window_size, 1)
            X = np.arange(len(q)).reshape(-1, 1).astype(np.int32) 

            linear_regressor_window = LinearRegression()
            linear_regressor_window.fit(X, q)
            slope_lin_fit_window[i] = linear_regressor_window.coef_[0]
            intercept_lin_fit_window[i] = linear_regressor_window.intercept_
            discharge_capacity_1[i] = q[0][0]
            diff_discharge_capacity_max_1[i] = np.max(q) - q[0][0]

            # 3. Other features
            mean_discharge_time[i] = np.mean(cell_summary['Discharge_time'][summary_key_c1:summary_key_clast+1])
            minimum_IR_window[i] = np.min(cell_summary['IR'][summary_key_c1:summary_key_clast+1])
            diff_IR_window[i] = cell_summary['IR'][summary_key_clast] - cell_summary['IR'][summary_key_c1]

            # 4. Targets
            target_remaining[i] = cell_summary['Remaining_cycles'][summary_key_clast]
            target_current[i] = int(key_clast)
            target_classifier[i] = cell_v['cycle_life'] >= 550

            
        # assemble cell-level df
        cell_dfs.append(
            pd.DataFrame({
                "cell_key": np.array(cell_k),
                "cell_batch": np.array(cell_batch),
                "cell_num": np.array(cell_num),
                "minimum_dQ_window": minimum_dQ_window,
                "variance_dQ_window": variance_dQ_window,
                "skewness_dQ_window": skewness_dQ_window,
                "kurtosis_dQ_window": kurtosis_dQ_window,
                "slope_lin_fit_window": slope_lin_fit_window,
                "intercept_lin_fit_window": intercept_lin_fit_window,
                "discharge_capacity_1": discharge_capacity_1,
                "diff_discharge_capacity_max_1": diff_discharge_capacity_max_1,
                "mean_discharge_time": mean_discharge_time,
                "minimum_IR_window": minimum_IR_window,
                "diff_IR_window": diff_IR_window,
                "target_remaining": target_remaining,
                "target_current": target_current,
                "target_classifier": target_classifier
            }))
    
    return pd.concat(cell_dfs)


if __name__ == "__main__":
    preprocessed_pkl = pickle.load(open(cst.PROCESSED_DATA, "rb"))  # dict
    windowed_features_df = build_windowed_feature_df(preprocessed_pkl, WINDOW_SIZE, W_SHIFT, W_STRIDE)
    fname = "rebuild_windowed_features_{}_{}_{}.csv".format(WINDOW_SIZE, W_SHIFT, W_STRIDE)
    save_csv_path = join(cst.DATA_DIR, fname)
    windowed_features_df.to_csv(save_csv_path, index=False)
    print("Saved features to", save_csv_path)