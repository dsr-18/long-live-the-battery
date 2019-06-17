from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np


def print_dict_keys(print_dict, a=0, ident=2, max_depth=100):
    """Prints all keys of a dictionary recursively up until the maximum depth.
    
    Arguments:
        print_dict {dict} -- The python dictionary to print.
    
    Keyword Arguments:
        a {int} -- Counter for the recursive calling of the print function. (default: {0})
        ident {int} -- The number of spaces to use to ident the different key levels. (default: {2})
        max_depth {int} -- The maximum depth in the print_dict to print the keys. (default: {100})
    """
    for key, value in print_dict.items():
        print(" " * a + f"[{key}]")

        if isinstance(value, dict) and max_depth > a / ident:
            print_dict_keys(value, a + ident, max_depth=max_depth)
            

def simple_plotly(x, inline=False, **kwargs):
    """Plots a simple plotly plot for all keyword arguments over x.
    Keayword arguments are used for naming the different traces."""
    
    pyo.init_notebook_mode(connected=True)
    traces = []
    
    for y_key, y_value in kwargs.items():
        traces.append(go.Scatter(dict(
            x=x, 
            y=y_value, 
            mode='lines+markers', 
            name=y_key
        )))
    
    fig = go.Figure(traces)
    fig['layout'].update(height=1000, width=1000)
    
    if inline:
        pyo.iplot(fig)
    else:
        pyo.plot(fig)
    

def debug_plot(Qd, T, V, t):
    """Conveniance function for debugging."""
    sample_space = np.arange(len(V))
    
    simple_plotly(sample_space, V=V, Q=Qd, T=T, t=t)
    simple_plotly(Qd, V=V)
    simple_plotly(T, V=V)
    

def plot_cycle_results(cycle_results_dict, inline=False):
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
            mode='markers', 
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
            mode='markers',
            name='T original data'
        )))
    traces2.append(go.Scatter(dict(
        x=cycle_results_dict["T_resample"],
        y=cycle_results_dict["V_resample"],
        mode='lines+markers',
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