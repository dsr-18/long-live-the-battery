import pickle
import warnings
from pprint import pprint

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from trainer.rebuilding_features import load_batches_to_dict
import trainer.constants as cst


class DropCycleException(Exception):
    """Used for dropping whole cycles without additional information."""
    pass


class OutlierException(Exception):
    """Used for dropping whole cycles based on detected outliers"""
    def __init__(self, message, outlier_dict):
        super().__init__(message)
        self.outlier_dict = outlier_dict


def check_for_drop_warning(array_before, array_after, drop_warning_thresh=0.10):
    """Checks weather the size of array_after is "drop_warning_thresh"-percent
    smaller than array_before and issues a warning in that case."""
    
    try:
        assert float(len(array_before) - len(array_after)) / len(array_before) < drop_warning_thresh, \
            """More than {} percent of values were dropped ({} out of {}).""".format(
                drop_warning_thresh * 100,
                len(array_before) - len(array_after),
                len(array_before))
    except AssertionError as e:
        warnings.warn(str(e))
        # simple_plotly(array_before[-1], V_original=array_before[2])
        # simple_plotly(array_after[-1], V_indexed=array_after[2])
    finally:
        pass


def multiple_array_indexing(valid_numpy_index, *args, drop_warning=False, drop_warning_thresh=0.10):
    """Indexes multiple numpy arrays at once and returns the result in a tuple.
    
    Arguments:
        numpy_index {numpy.ndarray or integer sequence} -- The used indeces.
    
    Returns:
        tuple -- reindexed numpy arrays from *args in the same order.
    """
    indexed_arrays = [arg[valid_numpy_index].copy() for arg in args]    
    
    if drop_warning:
        check_for_drop_warning(args[0], indexed_arrays[0])
    return tuple(indexed_arrays)


def outlier_dict_without_mask(outlier_dict):
    """Modifies an outlier dict for printing purposes by removing the mask. 
    
    Arguments:
        outlier_dict {dict} -- Original outliert dictionary.
    
    Returns:
        dict -- Same outlier dict without the key "outliert_mask"
    """
    outlier_dict_wo_mask = dict()
    for key in outlier_dict.keys():
        outlier_dict_wo_mask[key] = {k: v for k, v in outlier_dict[key].items() if k != "outlier_mask"}
    return outlier_dict_wo_mask


def compute_outlier_dict(std_multiple_threshold, verbose=False, **kwargs):
    """Checks for outliers in all numpy arrays given in kwargs by computing the standard deveation of np.diff().
    Outliers for every array are defined at the indeces, where the np.diff() is bigger than
    std_multiple_threshold times the standard deviation.
    
    Keyword Arguments:
        std_multiple_threshold {int} -- Threshold that defines an outlier by multiplying with the
            standard deveation (default: {15})
        verbose {bool} -- If True, prints the values for every found outlier (default: {False})
    
    Returns:
        dict -- The outliert results taged by the names given in kwargs
    """
    outlier_dict = dict()
    
    for key, value in kwargs.items():
        diff_values = np.diff(value, prepend=value[0])
        std_diff = diff_values.std()
        outlier_mask = diff_values > (std_multiple_threshold * std_diff)  # Get the mask for all outliers
        outlier_indeces = np.argwhere(outlier_mask)  # Get the indeces for all outliers
        
        if outlier_indeces.size > 0:  # Add outlier information to the outlier dict, if an outlier has been found
            outlier_dict[key] = dict(std_diff=std_diff,
                                     original_values=value[outlier_indeces],
                                     diff_values=diff_values[outlier_indeces],
                                     outlier_indeces=outlier_indeces,
                                     outlier_mask=outlier_mask)
    
    if verbose and outlier_dict:  
        # If outlier_dict has any entries, then print a version without the mask (too big for printing)
        outlier_dict_wo_mask = outlier_dict_without_mask(outlier_dict)  # Generate a smaller dict for better printing
        print("############ Found outliers ############ ")
        pprint(outlier_dict_wo_mask)
        print("")
    return outlier_dict


def drop_cycle_big_t_outliers(std_multiple_threshold, Qd, T, V, t, t_diff_outlier_thresh=100):
    """Checks for big outliers in the np.diff() values of t.
    If any are found the whole cyce is dropped, with one exception:
        There is only one outlier which lays right after the end of discharging.
        In this case, all measurement values of Qd, T, V and t after this outlier are dropped and their values returned.
    
        The end of discharging is defined as a V value below 2.01.
    
    Arguments:
        outlier_dict {dict} -- Dictionary with outlier information for the whole cycle.
        Qd {numpy.ndarray} -- Qd during discharging
        T {numpy.ndarray} -- T during discharging
        V {numpy.ndarray} -- V during discharging
        t {numpy.ndarray} -- t during discharging
        t_diff_outlier_thresh {int} -- Threshold that defines what a "big" t outliert is
    
    Raises:
        OutlierException: Will be raised, if the whole cycle should be dropped.
    
    Returns:
        Tuple of numpy.ndarray  -- Returns the original values of Qd, T, V and t if no big t outlier is found, or
            a slice of all arrays if the only outlier lays right after the end of discharging.
    """
    outlier_dict = compute_outlier_dict(std_multiple_threshold=std_multiple_threshold, Qd=Qd, T=T, V=V, t=t)
    if outlier_dict.get("t"):  # If any outliert was found in t
        t_outlier_mask = outlier_dict["t"]["diff_values"] > t_diff_outlier_thresh
    else:
        t_outlier_mask = None
    
    if np.any(t_outlier_mask):  # Only do something if there are big outliers.
        # Get the indeces 1 before the t outliers.
        indeces_before_t_outliers = outlier_dict["t"]["outlier_indeces"][t_outlier_mask] - 1
        # Get the minimum V value right before all t outliers.
        V_before_t_outlier = np.min(V[indeces_before_t_outliers])
        
        # If there is excatly one t outlier right at the end of discharging,
        #   drop all values after this index and continue with processing.
        if indeces_before_t_outliers.size == 1 and V_before_t_outlier < 2.01:
            i = int(indeces_before_t_outliers) + 1
            return Qd[:i], T[:i], V[:i], t[:i]
        else:
            raise OutlierException(
                "    Dropping cycle based on outliers with np.diff(t) > {} with value(s) {}".format(
                    t_diff_outlier_thresh,                    
                    list(outlier_dict["t"]["diff_values"][t_outlier_mask])),
                outlier_dict)
    else:
        return Qd, T, V, t


def drop_outliers_starting_left(std_multiple_threshold, Qd, T, V, t):
    """Searches for outliers in Qd, T, V and t and drops them one by one starting with the smallest index.
    Outlier indeces are dropped from every array simultaniously, so the sizes still match.
    After the first outliers from every array have been dropped, outliers are computed again, to not drop
    false detections.
    
    Arguments:
        std_multiple_threshold {int} -- Threshold for the compute_outlier_dict function
        Qd {numpy.ndarray} -- Qd measurements
        T {numpy.ndarray} -- T measurements
        V {numpy.ndarray} -- V measurements
        t {numpy.ndarray} -- t measurements
    
    Returns:
        tuple of numpy.ndarrays -- All arrays without outliers
    """
    Qd_, T_, V_, t_ = Qd.copy(), T.copy(), V.copy(), t.copy()
    
    # Initialize and compute outliers
    drop_counter = 0
    outlier_dict = compute_outlier_dict(std_multiple_threshold, verbose=True, Qd=Qd_, T=T_, V=V_, t=t_)
    original_outlier_dict = outlier_dict  # copy for debugging und raising OutlierException.
    
    # Process until no outliers are found.
    while outlier_dict:  
        # Get indeces of the left most outlier for every array.      
        first_outlier_indeces = [np.min(outlier_info["outlier_indeces"]) for outlier_info in outlier_dict.values()]
        # Only consider every index once and make it a list type for numpy indexing in array_exclude_index().
        unique_indeces_to_drop = list(set(first_outlier_indeces))
        
        # Drop all unique outlier indeces from all arrays.
        Qd_ = array_exclude_index(Qd_, unique_indeces_to_drop)
        T_ = array_exclude_index(T_, unique_indeces_to_drop)
        V_ = array_exclude_index(V_, unique_indeces_to_drop)
        t_ = array_exclude_index(t_, unique_indeces_to_drop)
        
        drop_counter += len(unique_indeces_to_drop)
        
        # Recompute outlierts after dropping the unique indeces from all arrays.
        outlier_dict = compute_outlier_dict(std_multiple_threshold, Qd=Qd_, T=T_, V=V_, t=t_)
    
    if drop_counter > 0:
        print("    Dropped {} outliers in {}".format(drop_counter, list(original_outlier_dict.keys())))
        print("")
    
    check_for_drop_warning(Qd, Qd_)
    return Qd_, T_, V_, t_


def array_exclude_index(arr, id):
    """Returns the given array without the entry at id.
    id can be any valid numpy index."""
    
    mask = np.ones_like(arr, bool)
    mask[id] = False
    return arr[mask]


def handle_small_Qd_outliers(std_multiple_threshold, Qd, t, Qd_max_outlier=0.06):
    """Handles specifically small outliers in Qd, which are a result of constant values for a
    small number of measurements before the "outlier". The constant values are imputed by linearly interpolating
    Qd over t, since Qd over t should be linear anyways. This way the "outlier" is "neutralized", since there is no
    "step" left from the constant values to the outlier value.
    
    Arguments:
        std_multiple_threshold {int} -- Threshold to use for the compute_outlier_dict function
        Qd {numpy.ndarray} -- Qd measurements
        t {numpy.ndarray} -- t measurements corresponding to Qd
    
    Keyword Arguments:
        Qd_max_outlier {float} -- The maximum absolute value for the found outliers in Qd, which get handled
            by this function.
        This is needed only to make the function more specific. (default: {0.06})
    
    Returns:
        numpy.ndarray -- The interpolated version of Qd.
    """
    
    Qd_ = Qd.copy()  # Only copy Qd, since it is the only array values are assigned to
    outlier_dict = compute_outlier_dict(std_multiple_threshold, Qd=Qd_)
    
    if outlier_dict.get("Qd"):
        # Get only the indeces of all small outliers
        small_diff_value_mask = outlier_dict["Qd"]["diff_values"] <= Qd_max_outlier
        ids = outlier_dict["Qd"]["outlier_indeces"][small_diff_value_mask]
    else:
        ids = None
    
    if ids:
        # Interpolate all values before small outliers that stay constant (np.diff == 0)
        for i in ids:
            # Get the last index, where the value of Qd doesn't stay constant before the outlier.
            start_id = int(np.argwhere(np.diff(Qd_[:i]) > 0)[-1])
            
            # Make a mask for where to interpolate
            interp_mask = np.zeros_like(Qd_, dtype=bool)
            interp_mask[start_id:i] = True
            interp_values = np.interp(
                t[interp_mask],  # Where to evaluate the interpolation function.
                t[~interp_mask],  # X values for the interpolation function.
                Qd_[~interp_mask]  # Y values for the interpolation function.
            )
            # Assign the interpolated values
            Qd_[interp_mask] = interp_values
            print("    Interpolated small Qd outlier from index {} to {}".format(start_id, i))
            
    return Qd_


def make_strictly_decreasing(x_interp, y_interp, prepend_value=3.7):     
    """Takes a monotonically decreasing array y_interp and makes it strictly decreasing by interpolation over x_interp.
    
    Arguments:
        x_interp {numpy.ndarray} -- The values to interpolate over.
        y_interp {numpy.ndarray} -- Monotonically decreasing values.
    
    Keyword Arguments:
        prepend_value {float} -- Value to prepend to y_interp for assesing the difference to the preceding value.
            (default: {3.7})
    
    Returns:
        numpy.ndarray -- y_interp with interpolated values, where there used to be zero difference to
            the preceding value.
    """
    y_interp_copy = y_interp.copy()
    # Make the tale interpolatable if the last value is not the single minimum.
    if y_interp_copy[-1] >= y_interp_copy[-2]:
        y_interp_copy[-1] -= 0.0001
        
    # Build a mask for all values, which do not decrease.
    bigger_equal_zero_diff = np.diff(y_interp_copy, prepend=prepend_value) >= 0
    # Replace these values with interpolations based on their border values.
    interp_values = np.interp(
        x_interp[bigger_equal_zero_diff],  # Where to evaluate the interpolation function.
        x_interp[~bigger_equal_zero_diff],  # X values for the interpolation function.
        y_interp_copy[~bigger_equal_zero_diff]  # Y values for the interpolation function.
    )
    y_interp_copy[bigger_equal_zero_diff] = interp_values
    
    # This has to be given, since interpolation will fail otherwise.
    assert np.all(np.diff(y_interp_copy) < 0), "The result y_copy is not strictly decreasing!"
                
    return y_interp_copy


def preprocess_cycle(cycle,
                     I_thresh=-3.99, 
                     Vdlin_start=3.5, 
                     Vdlin_stop=2.0, 
                     Vdlin_steps=cst.STEPS,
                     return_original_data=False):
    """Processes data (Qd, T, V, t) from one cycle and resamples Qd, T and V to a predefinded dimension.
    discharging_time will be computed based on t and is the only returned feature that is a scalar.
    
    Arguments:
        cycle {dict} -- One cycle entry from the original data with keys 'I', 'Qd', 'T', 'V', 't'
    
    Keyword Arguments:
        I_thresh {float} -- Only measurements where the current is smaller than this threshold are chosen
            (default: {-3.99})
        Vdlin_start {float} -- Start value for the resampled V (default: {3.5})
        Vdlin_stop {float} -- Stop value for the resampled V (default: {2.0})
        Vdlin_steps {int} -- Number of steps V, Qd and T are resampled (default: {1000})
        return_original_data {bool} -- Weather the original datapoints, which were used for interpolation,
            shold be returned in the results  (default: {False})
    
    Returns:
        {dict} -- Dictionary with the resampled (and original) values. 
    """

    Qd = cycle["Qd"]
    T = cycle["T"]
    V = cycle["V"]
    I = cycle["I"]  # noqa: E741
    t = cycle["t"]
    
    # Only take the measurements during high current discharging.
    discharge_mask = I < I_thresh
    Qd, T, V, t = multiple_array_indexing(discharge_mask, Qd, T, V, t)
    
    # Sort all values after time.
    sorted_indeces = t.argsort()
    Qd, T, V, t = multiple_array_indexing(sorted_indeces, Qd, T, V, t)
    
    # Only take timesteps where time is strictly increasing.
    increasing_time_mask = np.diff(t, prepend=0) > 0
    Qd, T, V, t = multiple_array_indexing(increasing_time_mask, Qd, T, V, t)

    # Dropping outliers.
    Qd, T, V, t = drop_cycle_big_t_outliers(15, Qd, T, V, t)
    
    Qd = handle_small_Qd_outliers(12, Qd, t)
    
    Qd, T, V, t = drop_outliers_starting_left(12, Qd, T, V, t)
    
    # Apply savitzky golay filter to V to smooth out the values.
    # This is done in order to not drop too many values in the next processing step (make monotonically decreasing).
    # This way the resulting curves don't become skewed too much in the direction of smaller values.
    savgol_window_length = 25
    if savgol_window_length >= V.size:
        raise DropCycleException("""Dropping cycle with less than {} V values.\nSizes --> Qd:{}, T:{}, V:{}, t:{}"""
                                 .format(savgol_window_length, Qd.size, T.size, V.size, t.size))
    V_savgol = savgol_filter(V, window_length=25, polyorder=2)

    # Only take the measurements, where V is monotonically decreasing (needed for interpolation).
    # This is done by comparing V to the accumulated minimum of V.
    #    accumulated minimum --> (taking always the smallest seen value from V from left to right)
    v_decreasing_mask = V_savgol == np.minimum.accumulate(V_savgol)
    Qd, T, V, t = multiple_array_indexing(v_decreasing_mask, Qd, T, V_savgol, t, drop_warning=True)
    
    # Make V_3 strictly decreasing (needed for interpolation).
    V_strict_dec = make_strictly_decreasing(t, V)

    # Calculate discharging time. (Only scalar feature which is returned later)
    discharging_time = t.max() - t.min()
    if discharging_time < 6:
        print("Test")
        raise DropCycleException("Dropping cycle with discharge_time = {}"
                                 .format(discharging_time))
    
    # Make interpolation function.
    Qd_interp_func = interp1d(
        V_strict_dec[::-1],  # V_strict_dec is inverted because it has to be increasing for interpolation.
        Qd[::-1],  # Qd and T are also inverted, so the correct values line up.
        bounds_error=False,  # Allows the function to be evaluated outside of the range of V_strict_dec.
        fill_value=(Qd[::-1][0], Qd[::-1][-1])  # Values to use, when evaluated outside of V_strict_dec.
    )
    T_interp_func = interp1d(
        V_strict_dec[::-1],
        T[::-1],
        bounds_error=False,
        fill_value=(T[::-1][0], T[::-1][-1])
    )

    # For resampling the decreasing order is chosen again.
    # The order doesn't matter for evaluating Qd_interp_func.
    Vdlin = np.linspace(Vdlin_start, Vdlin_stop, Vdlin_steps)
    
    Qdlin = Qd_interp_func(Vdlin)
    Tdlin = T_interp_func(Vdlin)

    if return_original_data:
        return {
            cst.QDLIN_NAME: Qdlin,
            cst.TDLIN_NAME: Tdlin,
            cst.VDLIN_NAME: Vdlin,
            cst.DISCHARGE_TIME_NAME: discharging_time,
            # Original data used for interpolation.
            "Qd_original_data": Qd,
            "T_original_data": T,
            "V_original_data": V,
            "t_original_data": t
        }
    else:
        return {
            cst.QDLIN_NAME: Qdlin,
            cst.TDLIN_NAME: Tdlin,
            cst.VDLIN_NAME: Vdlin,
            cst.DISCHARGE_TIME_NAME: discharging_time
        }


def preprocess_batch(batch_dict, return_original_data=False, return_cycle_drop_info=False, verbose=False):
    """Processes all cycles of every cell in batch_dict and returns the results in the same format.
    
    Arguments:
        batch_dict {dict} -- Unprocessed batch of cell data.
    
    Keyword Arguments:
        return_original_data {bool} -- If True, the original data used for interpolation is returned. (default: {False})
        verbose {bool} -- If True prints progress for every cell (default: {False})
    
    Returns:
        dict -- Results in the same format as batch_dict.
    """
    batch_results = dict()
    cycles_drop_info = dict()
    
    print("Start processing data ...")
    
    for cell_key in list(batch_dict.keys()):
        # The iteration is over a list of keys so the processed keys can be removed while iterating over the dict.
        # This reduces the memory used during processing.
        # If "for cell_key, cell_value in batch_dict.items()" is used,
        #    "del batch_dict[cell_key]" would throw an RuntimeError: dictionary changed size during iteration.
        cell_value = batch_dict[cell_key]
        # Initialite the cell results with all available scalar values.
        batch_results[cell_key] = dict(
            cycle_life=cell_value["cycle_life"][0][0],
            summary={
                cst.INTERNAL_RESISTANCE_NAME: [],
                cst.QD_NAME: [],
                cst.REMAINING_CYCLES_NAME: [],
                cst.DISCHARGE_TIME_NAME: []
            },
            cycles=dict()
        )
        
        for cycle_key, cycle_value in cell_value["cycles"].items():
            # Has to be skipped since there are often times only two measurements.
            if cycle_key == '0':
                continue
            # Some cells have more cycle measurements than recorded cycle_life.
            # The reamining cycles will be dropped.
            elif int(cycle_key) > int(cell_value["cycle_life"][0][0]):
                print("    Cell {} has more cycles than cycle_life ({}): Dropping remaining cycles {} to {}"
                      .format(cell_key,
                              cell_value["cycle_life"][0][0],
                              cycle_key,
                              max([int(k) for k in cell_value["cycles"].keys()])))
                break
            
            # Start processing the cycle.
            try:
                cycle_results = preprocess_cycle(cycle_value, return_original_data=return_original_data)
            
            except DropCycleException as e:
                print("cell:", cell_key, " cycle:", cycle_key)
                print(e)
                print("")
                # Documenting dropped cell and key
                drop_info = {cell_key: {cycle_key: None}}
                cycles_drop_info.update(drop_info)                
                continue
            
            except OutlierException as oe:  # Can be raised if preprocess_cycle, if an outlier is found.
                print("cell:", cell_key, " cycle:", cycle_key)
                print(oe)
                print("")
                # Adding outlier dict from Exception to the cycles_drop_info.
                drop_info = {
                    cell_key: {
                        cycle_key: outlier_dict_without_mask(oe.outlier_dict)}}
                cycles_drop_info.update(drop_info)
                continue
            
            # Copy summary values for this cycle into the results. 
            # I tried writing it into an initialized array, but then indeces of dropped cycles get skipped. 
            batch_results[cell_key]["summary"][cst.INTERNAL_RESISTANCE_NAME].append(
                cell_value["summary"][cst.INTERNAL_RESISTANCE_NAME][int(cycle_key)])
            batch_results[cell_key]["summary"][cst.QD_NAME].append(
                cell_value["summary"][cst.QD_NAME][int(cycle_key)])
            batch_results[cell_key]["summary"][cst.REMAINING_CYCLES_NAME].append(
                cell_value["cycle_life"][0][0] - int(cycle_key))
            
            # Append the calculated discharge time.
            # This is the only scalar results from preprocess_cycle
            batch_results[cell_key]["summary"][cst.DISCHARGE_TIME_NAME].append(
                cycle_results.pop(cst.DISCHARGE_TIME_NAME))
            
            # Write the results to the correct cycle key.
            batch_results[cell_key]["cycles"][cycle_key] = cycle_results
        
        # Convert lists of appended values to numpy arrays.
        for k, v in batch_results[cell_key]["summary"].items():
            batch_results[cell_key]["summary"][k] = np.array(v)
        
        if verbose:
            print(cell_key, "done")
        # Delete cell key from dict, to reduce used memory during processing.
        del batch_dict[cell_key]
    
    cycles_drop_info["number_distinct_cells"] = len(cycles_drop_info)
    cycles_drop_info["number_distinct_cycles"] = sum([len(value) for key, value in cycles_drop_info.items()
                                                      if key != "number_distinct_cells"])

    print("Done processing data.")
    if return_cycle_drop_info:
        return batch_results, cycles_drop_info
    else:
        return batch_results


def describe_results_dict(results_dict):
    """Prints summary statistics for all computed results over every single cycle.
    This might take a few seconds, since it has to search the whole dictionary for every vallue."""
    print("Collecting results data and printing results ...")
    describe_dict = dict()
    
    cycle_life_list = [cell["cycle_life"] for cell in results_dict.values()]
    
    describe_dict.update(dict(
        cycle_life=dict(
            max=np.max(cycle_life_list),
            min=np.min(cycle_life_list),
            mean=np.mean(cycle_life_list),
            std=np.std(cycle_life_list)
        )
    ))

    summary_results = dict()
    for k in [cst.INTERNAL_RESISTANCE_NAME,
              cst.QD_NAME,
              cst.REMAINING_CYCLES_NAME,
              cst.DISCHARGE_TIME_NAME]:
        summary_results[k] = dict(
            max=np.max([np.max(cell["summary"][k]) for cell in results_dict.values()]),
            min=np.min([np.min(cell["summary"][k]) for cell in results_dict.values()]),
            mean=np.mean([np.mean(cell["summary"][k]) for cell in results_dict.values()]),
            mean_std=np.std([np.std(cell["summary"][k]) for cell in results_dict.values()])
        )
    describe_dict.update(dict(summary_results=summary_results))
    
    cycle_results = dict()
    for k in [cst.QDLIN_NAME, cst.TDLIN_NAME]:
        cycle_results[k] = dict(
            max=np.max([np.max(cycle[k]) for cell in results_dict.values() for cycle in cell["cycles"].values()]),
            min=np.min([np.min(cycle[k]) for cell in results_dict.values() for cycle in cell["cycles"].values()]),
            mean=np.mean([np.mean(cycle[k]) for cell in results_dict.values() for cycle in cell["cycles"].values()]),
            mean_std=np.mean([np.std(cycle[k]) for cell in results_dict.values() for cycle in cell["cycles"].values()])
        )
    describe_dict.update(dict(cycle_results=cycle_results))
    
    pprint(describe_dict)


def save_preprocessed_data(results_dict, save_dir=cst.PROCESSED_DATA):
    print("Saving preprocessed data to {}".format(save_dir))
    with open(save_dir, 'wb') as f:
        pickle.dump(results_dict, f)
    

def load_preprocessed_data(save_dir=cst.PROCESSED_DATA):
    print("Loading preprocessed data from {}".format(save_dir))
    with open(save_dir, 'rb') as f:
        return pickle.load(f)


def main():
    batch_dict = load_batches_to_dict(amount_to_load=3)    

    results, cycles_drop_info = preprocess_batch(batch_dict,
                                                 return_original_data=False,
                                                 return_cycle_drop_info=True,
                                                 verbose=True)
    
    pprint(cycles_drop_info)
    # describe_results_dict(results)
    
    save_preprocessed_data(results)
    print("Done!")
   
 
if __name__ == "__main__":
    main()


# TODO: Check with new threshold after processing outliers of std >= 12
# TODO: Notebook update