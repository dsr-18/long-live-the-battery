import plotly.offline as pyo
import plotly.graph_objs as go


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
        print(" "*a + f"[{key}]")

        if isinstance(value, dict) and max_depth > a/ident:
            print_dict_keys(value, a+ident, max_depth=max_depth)
            

def simple_plotly(x, **kwargs):
    
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
    pyo.plot(fig)