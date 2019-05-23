import numpy as np
from scipy.interpolate import interp1d
from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go

def preprocess_cycle(
    Qd, 
    T, 
    V, 
    I, 
    I_thresh=-3.99, 
    V_resample_start=3.5, 
    V_resample_stop=2.0, 
    V_resample_steps=1000,
    return_original_data=False):
    
    ## Only take the measurements during high current discharging.
    high_current_discharge = I < I_thresh
    
    Qd_dis = Qd[high_current_discharge]
    T_dis = T[high_current_discharge]
    V_dis = V[high_current_discharge]    
    
    ## Only take the measurements, where V is decreasing (needed for interpolation).
    # This is done by comparing V_dis to the accumulated minimum of V_dis.
    #    accumulated minimum --> (taking always the smallest seen value from V_dis from left to right)
    # Then only the values that actually are the smallest value up until their index are chosen.
    v_decreasing = V_dis == np.minimum.accumulate(V_dis)

    Qd_dis_dec = Qd_dis[v_decreasing]
    T_dis_dec = T_dis[v_decreasing]
    V_dis_dec = V_dis[v_decreasing]
    
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
    
    # Quick check before interpolating.
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
            Qd_original_data=Qd_dis_dec,   # Also return the subset of the original data for later comparison.
            T_original_data=T_dis_dec,
            V_original_data=V_dis_dec
            )
    else:
        return dict(
            Qd_resample=Qd_resample,
            T_resample=T_resample,
            V_resample=V_resample
            )

def plot_preprocessing_results(results_dict):
    """Plots comparison curves with plotly for a results dict return from preprocess_cycle_curves"""

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

    #fig['layout'].update(height=1000, width=1000)
    pyo.plot(fig)