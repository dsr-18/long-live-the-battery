import numpy as np
import pandas as pd
import pickle
from os.path import join

# TODO consolidate constants & filepaths throughout codebase
DATA_DIR = join("data")

def load_batches_to_dict(amount_to_load=3):
    """Loads batches from disc and returns one concatenated dict.
    amount_to_load specifies the number of batches to load, starting from 1."""
    if amount_to_load < 1 or amount_to_load > 3:
        raise "amount_to_load is not a valid number! Try a number between 1 and 3."

    batches_dict = {}  # Initializing

    # Replicating Load Data logic
    print("Loading batch1 ...")
    path1 = join(DATA_DIR, "batch1.pkl")
    batch1 = pickle.load(open(path1, 'rb'))

    #remove batteries that do not reach 80% capacity
    del batch1['b1c8']
    del batch1['b1c10']
    del batch1['b1c12']
    del batch1['b1c13']
    del batch1['b1c22']

    batches_dict.update(batch1)

    if amount_to_load > 1:
        print("Loading batch2 ...")
        path2 = join(DATA_DIR, "batch2.pkl")
        batch2 = pickle.load(open(path2, 'rb'))

        # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
        # and put it with the correct cell from batch1
        batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
        batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
        add_len = [662, 981, 1060, 208, 482]

        for i, bk in enumerate(batch1_keys):
            batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
            for j in batch1[bk]['summary'].keys():
                if j == 'cycle':
                    batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
                else:
                    batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
            last_cycle = len(batch1[bk]['cycles'].keys())
            for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
                batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]
        
        del batch2['b2c7']
        del batch2['b2c8']
        del batch2['b2c9']
        del batch2['b2c15']
        del batch2['b2c16']

        # All keys have to be updated after the reordering.
        batches_dict.update(batch1)
        batches_dict.update(batch2)

    if amount_to_load > 2:
        print("Loading batch3 ...")
        path3 = join(DATA_DIR, "batch3.pkl")
        batch3 = pickle.load(open(path3, 'rb'))

        # remove noisy channels from batch3
        del batch3['b3c37']
        del batch3['b3c2']
        del batch3['b3c23']
        del batch3['b3c32']
        del batch3['b3c38']
        del batch3['b3c39']

        batches_dict.update(batch3)

    print("Done loading batches")
    return batches_dict


def build_feature_df(batch_dict):
    """Returns a pandas DataFrame with all originally used features out of a loaded batch dict"""

    print("Start building features ...")
    
    from scipy.stats import skew, kurtosis
    from sklearn.linear_model import LinearRegression
    
    n_cells = len(batch_dict.keys())

    ## Initializing feature vectors:
    cycle_life = np.zeros(n_cells)
    # 1. delta_Q_100_10(V)
    minimum_dQ_100_10 = np.zeros(n_cells)
    variance_dQ_100_10 = np.zeros(n_cells)
    skewness_dQ_100_10 = np.zeros(n_cells)
    kurtosis_dQ_100_10 = np.zeros(n_cells)

    # 2. Discharge capacity fade curve features
    slope_lin_fit_2_100 = np.zeros(n_cells)  # Slope of the linear fit to the capacity fade curve, cycles 2 to 100
    intercept_lin_fit_2_100 = np.zeros(n_cells)  # Intercept of the linear fit to capavity face curve, cycles 2 to 100
    discharge_capacity_2 = np.zeros(n_cells)  # Discharge capacity, cycle 2
    diff_discharge_capacity_max_2 = np.zeros(n_cells)  # Difference between max discharge capacity and cycle 2

    # 3. Other features
    mean_charge_time_2_6 = np.zeros(n_cells)  # Average charge time, cycle 2 to 6
    minimum_IR_2_100 = np.zeros(n_cells)  # Minimum internal resistance
    # I skipped the tempererature integreal because i have no idea what it means.
    diff_IR_100_2 = np.zeros(n_cells)  # Internal resistance, difference between cycle 100 and cycle 2

    # Classifier features
    minimum_dQ_5_4 = np.zeros(n_cells)
    variance_dQ_5_4 = np.zeros(n_cells)
    cycle_550_clf = np.zeros(n_cells)

    for i, cell in enumerate(batch_dict.values()):
        cycle_life[i] = cell['cycle_life']

        # 1. delta_Q_100_10(V)    
        c10 = cell['cycles']['10']
        c100 = cell['cycles']['100']
        dQ_100_10 = c100['Qdlin'] - c10['Qdlin']
        
        minimum_dQ_100_10[i] = np.log(np.abs(np.min(dQ_100_10)))
        variance_dQ_100_10[i] = np.log(np.var(dQ_100_10))
        skewness_dQ_100_10[i] = np.log(np.abs(skew(dQ_100_10)))
        kurtosis_dQ_100_10[i] = np.log(np.abs(kurtosis(dQ_100_10)))

        # 2. Discharge capacity fade curve features
        # Compute linear fit for cycles 2 to 100:
        q = cell['summary']['QD'][1:100].reshape(-1, 1)  # discharge cappacities; q.shape = (99, 1); 
        X = cell['summary']['cycle'][1:100].reshape(-1, 1)  # Cylce index from 2 to 100; X.shape = (99, 1)
        linear_regressor_2_100 = LinearRegression()
        linear_regressor_2_100.fit(X, q)
        
        slope_lin_fit_2_100[i] = linear_regressor_2_100.coef_[0]
        intercept_lin_fit_2_100[i] = linear_regressor_2_100.intercept_
        discharge_capacity_2[i] = q[0][0]
        diff_discharge_capacity_max_2[i] = np.max(q) - q[0][0]
        
        # 3. Other features
        mean_charge_time_2_6[i] = np.mean(cell['summary']['chargetime'][1:6])
        minimum_IR_2_100[i] = np.min(cell['summary']['IR'][1:100])
        diff_IR_100_2[i] = cell['summary']['IR'][100] - cell['summary']['IR'][1]

        # Classifier features
        c4 = cell['cycles']['4']
        c5 = cell['cycles']['5']
        dQ_5_4 = c5['Qdlin'] - c4['Qdlin']
        minimum_dQ_5_4[i] = np.log(np.abs(np.min(dQ_5_4)))
        variance_dQ_5_4[i] = np.log(np.var(dQ_5_4))
        cycle_550_clf[i] = cell['cycle_life'] >= 550
        

    features_df = pd.DataFrame({
        "cell_key": np.array(list(batch_dict.keys())),
        "minimum_dQ_100_10": minimum_dQ_100_10,
        "variance_dQ_100_10": variance_dQ_100_10,
        "skewness_dQ_100_10": skewness_dQ_100_10,
        "kurtosis_dQ_100_10": kurtosis_dQ_100_10,
        "slope_lin_fit_2_100": slope_lin_fit_2_100,
        "intercept_lin_fit_2_100": intercept_lin_fit_2_100,
        "discharge_capacity_2": discharge_capacity_2,
        "diff_discharge_capacity_max_2": diff_discharge_capacity_max_2,
        "mean_charge_time_2_6": mean_charge_time_2_6,
        "minimum_IR_2_100": minimum_IR_2_100,
        "diff_IR_100_2": diff_IR_100_2,
        "minimum_dQ_5_4": minimum_dQ_5_4,
        "variance_dQ_5_4": variance_dQ_5_4,
        "cycle_life": cycle_life,
        "cycle_550_clf": cycle_550_clf
    })

    print("Done building features")
    return features_df

if __name__ == "__main__":
    all_batches_dict = load_batches_to_dict()
    features_df = build_feature_df(all_batches_dict)

    save_csv_path = join(DATA_DIR, "rebuild_features.csv")
    features_df.to_csv(save_csv_path, index=False)
    print("Saved features to ", save_csv_path)
