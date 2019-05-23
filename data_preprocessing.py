import numpy as np
from scipy.interpolate import interp1d
from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go


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
        cycle {batch["cell"]["cycles"]["cycle"]} -- One cycle entry from the original data.
    
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
    
    Qd_dis = Qd[high_current_discharge]
    T_dis = T[high_current_discharge]
    V_dis = V[high_current_discharge]
    t_dis = t[high_current_discharge]
    
    ## Only take the measurements, where V is decreasing (needed for interpolation).
    # This is done by comparing V_dis to the accumulated minimum of V_dis.
    #    accumulated minimum --> (taking always the smallest seen value from V_dis from left to right)
    # Then only the values that actually are the smallest value up until their index are chosen.
    v_decreasing = V_dis == np.minimum.accumulate(V_dis)

    Qd_dis_dec = Qd_dis[v_decreasing]
    T_dis_dec = T_dis[v_decreasing]
    V_dis_dec = V_dis[v_decreasing]
    t_dis_dec = t_dis[v_decreasing]

    high_current_discharging_time = t_dis_dec.max() - t_dis_dec.min()

    assert float(len(Qd_dis_dec))/len(Qd_dis) >= 0.95, \
        """More than 5 precent of values V_dis were dropped ({} out of {}).
        There might be a small outlier in V_dis.""".format(len(Qd_dis)-len(Qd_dis_dec), len(Qd_dis))
    
    ## Make V_dis_dec strictly decending (needed for interpolation).
    # Make a mask for only the V_dis_dec values that don't have zero difference to the preceding value.
    no_zero_diff = (np.diff(V_dis_dec, prepend=0) != 0)
    # Get the minimum absolute difference by which V_dis_dec is decreasing.
    min_diff = np.min(np.abs(np.diff(V_dis_dec[no_zero_diff])))
    # Substract half of the minimum difference to the values of V_dis_dec, which were not decreasing before.
    # This makes V_dis_inc strictly monotone.
    # Only half is substracted so that no new "zero diff" positions are created.
    V_dis_strict_dec = V_dis_dec - (~no_zero_diff * min_diff / 2)
    
    # Check before interpolating.
    assert np.all(np.diff(V_dis_strict_dec) < 0), "The result of V is not strictly decreasing. Do something."
    
    # Interpolate values and chose extrapolation, so interp_func can be evaluated over the whole v_resample range.
    # V_dis_strict_dec is inverted because it has to be increasing for interpolation.
    # Qd_dis_dec and T_dis_dec are also inverted, so the correct values line up.
    Qd_interp_func = interp1d(V_dis_strict_dec[::-1], Qd_dis_dec[::-1], fill_value='extrapolate')
    T_interp_func = interp1d(V_dis_strict_dec[::-1], T_dis_dec[::-1], fill_value='extrapolate')

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
            Qd_original_data=Qd_dis_dec,
            T_original_data=T_dis_dec,
            V_original_data=V_dis_dec,
            t_original_data=t_dis_dec
            )
    else:
        return dict(
            Qd_resample=Qd_resample,
            T_resample=T_resample,
            V_resample=V_resample,
            high_current_discharging_time=high_current_discharging_time
            )


def plot_preprocessing_results(results_dict):
    """Plots comparison curves with plotly for a results dict.
    
    Arguments:
        results_dict {dict} -- results returned from preprocess_cycle
    """

    pyo.init_notebook_mode(connected=True)

    Qd_original_data_trace = go.Scatter(dict(
        x=results_dict["Qd_original_data"], 
        y=results_dict["V_original_data"], 
        mode = 'markers', 
        name='Qd original data'
        ))
    Qd_resample_trace = go.Scatter(dict(
        x=results_dict["Qd_resample"], 
        y=results_dict["V_resample"], 
        mode='lines+markers', 
        name='Qd resampled'
        ))

    T_original_data_trace = go.Scatter(dict(
        x=results_dict["T_original_data"],
        y=results_dict["V_original_data"],
        mode = 'markers',
        name='T original data'
        ))
    T_resample_trace = go.Scatter(dict(
        x=results_dict["T_resample"],
        y=results_dict["V_resample"],
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