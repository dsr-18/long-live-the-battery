import numpy as np
from scipy.interpolate import interp1d
from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go
import warnings


def multiple_array_indexing(valid_numpy_index, *args, drop_warning=False, drop_warning_thresh=0.10):
    """Indexes multiple numpy arrays at once and returns the result in a tuple.
    
    Arguments:
        numpy_index {numpy.ndarray or integer sequence} -- The used indeces.
    
    Returns:
        tuple -- reindexed numpy arrays from *args in the same order.
    """
    indexed_arrays = [arg[valid_numpy_index].copy() for arg in args]    
    
    if drop_warning:
        # Check if a higher percentage of values than drop_warning_thresh were dropped during indexing.
        try:
            assert float(len(args[0])-len(indexed_arrays[0])) / len(args[0]) < drop_warning_thresh, \
                """More than {} percent of values were dropped ({} out of {}).""".format(
                        drop_warning_thresh*100,
                        len(args[0])-len(indexed_arrays[0]),
                        len(args[0])
                    )
        except AssertionError as e:
            warnings.warn(str(e))
            # from generic_helpers import simple_plotly
            # simple_plotly(args[-1], V_indexed=indexed_arrays[2], V_original=args[2])
        finally:
            return tuple(indexed_arrays)
    else:
        return tuple(indexed_arrays)


def outlier_dict_without_mask(outlier_dict):
    outlier_dict_wo_mask = dict()  # Generate a smaller dict for better printing
    for key in outlier_dict.keys():
        outlier_dict_wo_mask[key] = {k: v for k, v in outlier_dict[key].items() if k != "outlier_mask"}
    return outlier_dict_wo_mask


def check_outliers(std_multiple_threshold=15, verbose=False, **kwargs):
    """Checks for outliers in all numpy arrays given in kwargs by computing the standard deveation of np.diff().
    Outliers for every array are defined at the indeces, where the np.diff() is bigger than
    std_multiple_threshold times the standard deviation.
    
    Keyword Arguments:
        std_multiple_threshold {int} -- Threshold that defines an outlier (default: {15})
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
            outlier_dict[key] = dict(std_diff = std_diff,
                                     original_values = value[outlier_indeces],
                                     diff_values = diff_values[outlier_indeces],
                                     outlier_indeces = outlier_indeces,
                                     outlier_mask=outlier_mask)
    
    if verbose and outlier_dict:  
        # If outlier_dict has any entries, then print a version without the mask (too big for printing)
        outlier_dict_wo_mask = outlier_dict_without_mask(outlier_dict) # Generate a smaller dict for better printing
        print("#########################")
        print("Found outliers:")
        pprint(outlier_dict_wo_mask)
    return outlier_dict

def debug_plot(Qd, T, V, t):
    from generic_helpers import simple_plotly
    sample_space = np.arange(len(V))
    simple_plotly(sample_space, V=V, Q=Qd, T=T, t=t)
    simple_plotly(Qd, V=V)
    simple_plotly(T, V=V)


def drop_cycle_big_t_outliers(outlier_dict, Qd, T, V, t):
    if outlier_dict.get("t"):
        t_outlier_mask = outlier_dict["t"]["diff_values"] > 100
        if np.any(t_outlier_mask):  # Only do something if there are big outliers.
            # If the big t outlier is right at the end if discharging, the whole cycle should be dropped.
            indeces_before_t_outliers = outlier_dict["t"]["outlier_indeces"][t_outlier_mask] - 1  # Get the indeces 1 before the t outliers.
            V_before_t_outlier = np.min(V[indeces_before_t_outliers])  # Get the minimum V value right before all t outliers.
            
            # If there is excatly one t outlier right at the end of discharging, drop all values after this index and continue with processing.
            if indeces_before_t_outliers.size == 1 and V_before_t_outlier < 2.01:
                i = int(indeces_before_t_outliers) + 1
                return Qd[:i], T[:i], V[:i], t[:i]
            else:
                raise OutlierException(
                    "Dropping cycle based on t outliers with value(s) {}".format(list(outlier_dict["t"]["diff_values"][t_outlier_mask])),
                    outlier_dict)
    else:
        return Qd, T, V, t


def make_strictly_decreasing(x_interp, y_interp, prepend_value=3.7):     
    """Takes a monotonically decreasing array y_interp and makes it strictly decreasing by interpolation over x_interp.
    
    Arguments:
        x_interp {numpy.ndarray} -- The values to interpolate over.
        y_interp {numpy.ndarray} -- Monotonically decreasing values.
    
    Keyword Arguments:
        prepend_value {float} -- Value to prepend to y_interp for assesing the difference to the preceding value. (default: {3.7})
    
    Returns:
        numpy.ndarray -- y_interp with interpolated values, where there used to be zero difference to the preceding value.
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


def preprocess_cycle(
    cycle,
    I_thresh=-3.99, 
    V_resample_start=3.5, 
    V_resample_stop=2.0, 
    V_resample_steps=1000,
    return_original_data=False
    ):
    """Processes data (Qd, T, V, t) from one cycle and resamples Qd, T and V to a predefinded dimension.
    high_current_discharging_time will be computed based on t and is the only returned feature that is a scalar.
    
    Arguments:
        cycle {dict} -- One cycle entry from the original data with keys 'I', 'Qd', 'T', 'V', 't'
    
    Keyword Arguments:
        I_thresh {float} -- Only measurements where the current is smaller than this threshold are chosen (default: {-3.99})
        V_resample_start {float} -- Start value for the resampled V (default: {3.5})
        V_resample_stop {float} -- Stop value for the resampled V (default: {2.0})
        V_resample_steps {int} -- Number of steps V, Qd and T are resampled (default: {1000})
        return_original_data {bool} -- Weather the original datapoints, which were used for interpolation,
            shold be returned in the results  (default: {False})
    
    Returns:
        {dict} -- Dictionary with the resampled (and original) values. 
    """

    Qd = cycle["Qd"]
    T = cycle["T"]
    V = cycle["V"]
    I = cycle["I"]
    t = cycle["t"]
    
    ## Only take the measurements during high current discharging.
    high_current_discharge = I < I_thresh
    Qd, T, V, t = multiple_array_indexing(high_current_discharge, Qd, T, V, t)
    
    ## Sort all values after time.
    sort_indeces = t.argsort()
    Qd, T, V, t = multiple_array_indexing(sort_indeces, Qd, T, V, t)
    
    ## Only take timesteps where time is strictly increasing.
    increasing_time = np.diff(t, prepend=0) > 0
    Qd, T, V, t = multiple_array_indexing(increasing_time, Qd, T, V, t)

    ## Only take the measurements, where V is monotonically decreasing (needed for interpolation).
    # This is done by comparing V to the accumulated minimum of V.
    #    accumulated minimum --> (taking always the smallest seen value from V from left to right)
    
    outlier_dict = check_outliers(std_multiple_threshold=15, verbose=True, Qd=Qd, T=T, V=V, t=t)
    Qd, T, V, t = drop_cycle_big_t_outliers(outlier_dict, Qd, T, V, t)
    
    high_current_discharging_time = t.max() - t.min()  # Scalar feature which is returned later.
    if outlier_dict.get("t"):  # If an outlier was found, then calculate new discharge time.
        not_t_outliers = ~outlier_dict["t"]["outlier_mask"]
        high_current_discharging_time = np.sum(np.diff(t, prepend=t[0])[not_t_outliers])  # Only sum the diff values, that aren't a diff outlier.
    
    if outlier_dict:  # If V was an outlier before, check out the result
        debug_plot(Qd, T, V, t)

    v_decreasing = V == np.minimum.accumulate(V)
    Qd, T, V, t = multiple_array_indexing(v_decreasing, Qd, T, V, t, drop_warning=True)
    
    #check_outliers(Qd=Qd, T=T, V=V, t=t)
    if outlier_dict:  # If V was an outlier before, check out the result
        debug_plot(Qd, T, V, t)

    ## Make V_3 strictly decreasing (needed for interpolation).
    V_strict_dec = make_strictly_decreasing(t, V)

    if outlier_dict:  # If V was an outlier before, check out the result
        debug_plot(Qd, T, V, t)
  
        
    ## Make interpolation function.
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
    V_resample = np.linspace(V_resample_start, V_resample_stop, V_resample_steps)
    
    Qd_resample = Qd_interp_func(V_resample)
    T_resample = T_interp_func(V_resample)

    if return_original_data:
        return dict(
            Qd_resample=Qd_resample,
            T_resample=T_resample,
            V_resample=V_resample,
            high_current_discharging_time=high_current_discharging_time,
            # Original data used for interpolation.
            Qd_original_data=Qd,
            T_original_data=T,
            V_original_data=V,
            t_original_data=t
            )
    else:
        return dict(
            Qd_resample=Qd_resample,
            T_resample=T_resample,
            V_resample=V_resample,
            high_current_discharging_time=high_current_discharging_time
            )


def preprocess_batch(batch_dict, return_original_data=False, verbose=False):
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
    for cell_key, cell_value in batch_dict.items():
        batch_results[cell_key] = dict(
            cycle_life=cell_value["cycle_life"][0][0],
            summary=dict(
                IR = cell_value["summary"]["IR"],
                QD = cell_value["summary"]["QD"],
                remaining_cycle_life = cell_value["cycle_life"][0][0] - cell_value["summary"]["cycle"],
                high_current_discharging_time = np.zeros(int(cell_value["cycle_life"][0][0]))
            ),
            cycles=dict()
        )
        for cycle_key, cycle_value in cell_value["cycles"].items():
            if cycle_key == '0':  # Has to be skipped since there are often times only two measurements.
                continue
            cycle_results = preprocess_cycle(cycle_value, return_original_data=return_original_data)
            
            batch_results[cell_key]["summary"]["high_current_discharging_time"][int(cycle_key)-1] = \
                cycle_results.pop("high_current_discharging_time")
            batch_results[cell_key]["cycles"][cycle_key] = cycle_results
        if verbose:
            print("Processed", cell_key)
    
    return batch_results


def plot_preprocessing_results(cycle_results_dict, inline=True):
    """Plots comparison curves with plotly for a results dict of one cycle.
    When the original data is not included, only the resampled data is shown.
        
    Arguments:
        results_dict {dict} -- results returned from preprocess_cycle.
    """

    pyo.init_notebook_mode(connected=True)

    traces1 = []
    if "Qd_original_data" in cycle_results_dict.keys():
        traces1.append(go.Scatter(dict(
            x=cycle_results_dict["Qd_original_data"], 
            y=cycle_results_dict["V_original_data"], 
            mode = 'markers', 
            name='Qd original data'
            )))
    traces1.append(go.Scatter(dict(
        x=cycle_results_dict["Qd_resample"], 
        y=cycle_results_dict["V_resample"], 
        mode='lines+markers', 
        name='Qd resampled'
        )))
    
    traces2 = []
    if "T_original_data" in cycle_results_dict.keys():
        traces2.append(go.Scatter(dict(
            x=cycle_results_dict["T_original_data"],
            y=cycle_results_dict["V_original_data"],
            mode = 'markers',
            name='T original data'
            )))
    traces2.append(go.Scatter(dict(
        x=cycle_results_dict["T_resample"],
        y=cycle_results_dict["V_resample"],
        mode= 'lines+markers',
        name='T resampled'
        )))

    fig = tools.make_subplots(rows=2, cols=1)

    for trace in traces1:
        fig.append_trace(trace, 1, 1)

    for trace in traces2:
        fig.append_trace(trace, 2, 1)

    fig['layout'].update(height=1000, width=1000)
    
    if inline:
        pyo.iplot(fig)
    else:
        pyo.plot(fig)


def main():
    from rebuilding_features import load_batches_to_dict
    
    batch1 = load_batches_to_dict(amount_to_load=1)    

    results = preprocess_batch(batch1, return_original_data=True, verbose=True)
    print("Done!")

if __name__ == "__main__":
    main()