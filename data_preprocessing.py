import numpy as np
from scipy.interpolate import interp1d
from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go


def interpolate_zero_diff_values(x, y, prepend_value=3.7):
    y_copy = y.copy()
    # Build a mask for all values, which do not decrease.
    bigger_equal_zero_diff = np.diff(y_copy, prepend=prepend_value) >= 0
    # Replace these values with interpolations.
    interp_values = np.interp(
        x[bigger_equal_zero_diff],  # Where to evaluate the interpolation function.
        x[~bigger_equal_zero_diff],  # X values for the interpolation function.
        y_copy[~bigger_equal_zero_diff]  # Y values for the interpolation function.
        )
    y_copy[bigger_equal_zero_diff] = interp_values
    # If the last value has zero diff, the interpolation will replace this index with the same value.
    # In this case a small value will be subtracted.
    if bigger_equal_zero_diff[-1]:
        y_copy[-1] -= 0.000001

    assert np.all(np.diff(y_copy) < 0), "The result y_copy is not strictly decreasing. Do something."

    return y_copy


def preprocess_cycle(
    cycle,
    I_thresh=-3.99, 
    V_resample_start=3.5, 
    V_resample_stop=2.0, 
    V_resample_steps=1000,
    return_original_data=False):
    """Processes data (Qd, T, V, t) from one cycle and resamples Qd, T and V to a predefinded dimension.
    high_current_discharging_time will be computed with t and is the only returned feature that is a scalar.
    
    Arguments:
        cycle {dict} -- One cycle entry from the original data with keys 'I', 'Qd', 'T', 'V', 't'
    
    Keyword Arguments:
        I_thresh {float} -- Only measurements where the current is smaller than this threshold are chosen (default: {-3.99})
        V_resample_start {float} -- Start value for the resampled V (default: {3.5})
        V_resample_stop {float} -- Stop value for the resampled V (default: {2.0})
        V_resample_steps {int} -- Number of steps V (and Qd and T) are resampled (default: {1000})
        return_original_data {bool} -- Weather the original datapoints, which were used for interpolation,
            shold be returned in the results  (default: {False})
    
    Returns:
        {dict} -- Dictionary with the resampled values Qd_resample, T_resample over V_resample. 
    """

    Qd = cycle["Qd"]
    T = cycle["T"]
    V = cycle["V"]
    I = cycle["I"]
    t = cycle["t"]
    
    ## Only take the measurements during high current discharging.
    high_current_discharge = I < I_thresh
    
    Qd_1 = Qd[high_current_discharge]
    T_1 = T[high_current_discharge]
    V_1 = V[high_current_discharge]
    t_1 = t[high_current_discharge]
    
    ## Sort all values after time.
    sort_indeces = t_1.argsort()

    Qd_2 = Qd_1[sort_indeces]
    T_2 = T_1[sort_indeces]
    V_2 = V_1[sort_indeces]
    t_2 = t_1[sort_indeces]

    ## Only take the measurements, where V is decreasing (needed for interpolation).
    # This is done by comparing V_2 to the accumulated minimum of V_2.
    #    accumulated minimum --> (taking always the smallest seen value from V_2 from left to right)
    # Then only the values that actually are the smallest value up until their index are chosen.
    v_decreasing = V_2 == np.minimum.accumulate(V_2)

    Qd_3 = Qd_2[v_decreasing]
    T_3 = T_2[v_decreasing]
    V_3 = V_2[v_decreasing]
    t_3 = t_2[v_decreasing]

    high_current_discharging_time = t_3.max() - t_3.min()

    try:
        assert float(len(Qd_3))/len(Qd_2) >= 0.95, \
            """More than 5 precent of values were dropped ({} out of {}).
            There might be a small outlier in V_2.""".format(len(Qd_2)-len(Qd_3), len(Qd_2))
    except AssertionError as e:
        print(e)
        import pdb; pdb.set_trace()
        
    ## Make V_3 strictly decending (needed for interpolation).
    V_3_strict_dec = interpolate_zero_diff_values(t_3, V_3)

    # Interpolate values and chose extrapolation, so interp_func can be evaluated over the whole v_resample range.
    # V_3_strict_dec is inverted because it has to be increasing for interpolation.
    # Qd_3 and T_3 are also inverted, so the correct values line up.
    Qd_interp_func = interp1d(V_3_strict_dec[::-1], Qd_3[::-1], fill_value='extrapolate')
    T_interp_func = interp1d(V_3_strict_dec[::-1], T_3[::-1], fill_value='extrapolate')

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
            Qd_original_data=Qd_3,
            T_original_data=T_3,
            V_original_data=V_3,
            t_original_data=t_3
            )
    else:
        return dict(
            Qd_resample=Qd_resample,
            T_resample=T_resample,
            V_resample=V_resample,
            high_current_discharging_time=high_current_discharging_time
            )


def plot_preprocessing_results(cycle_results_dict):
    """Plots comparison curves with plotly for a results dict.
    
    Arguments:
        results_dict {dict} -- results returned from preprocess_cycle
    """

    pyo.init_notebook_mode(connected=True)

    Qd_original_data_trace = go.Scatter(dict(
        x=cycle_results_dict["Qd_original_data"], 
        y=cycle_results_dict["V_original_data"], 
        mode = 'markers', 
        name='Qd original data'
        ))
    Qd_resample_trace = go.Scatter(dict(
        x=cycle_results_dict["Qd_resample"], 
        y=cycle_results_dict["V_resample"], 
        mode='lines+markers', 
        name='Qd resampled'
        ))

    T_original_data_trace = go.Scatter(dict(
        x=cycle_results_dict["T_original_data"],
        y=cycle_results_dict["V_original_data"],
        mode = 'markers',
        name='T original data'
        ))
    T_resample_trace = go.Scatter(dict(
        x=cycle_results_dict["T_resample"],
        y=cycle_results_dict["V_resample"],
        mode= 'lines+markers',
        name='T resampled'
        ))

    fig = tools.make_subplots(rows=2, cols=1)

    fig.append_trace(Qd_resample_trace, 1, 1)
    fig.append_trace(Qd_original_data_trace, 1, 1)

    fig.append_trace(T_resample_trace, 2, 1)
    fig.append_trace(T_original_data_trace, 2, 1)

    fig['layout'].update(height=1000, width=1000)
    pyo.iplot(fig)