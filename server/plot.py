import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np

REMAINING_CYCLES_NAME = "Remaining_cycles"

def skewed_normalized_sigmoid(x):  
    sigmoid_mod = np.exp(-np.logaddexp(0, (-x + 5) * 1.2))
    return (sigmoid_mod - sigmoid_mod.min()) / (sigmoid_mod.max() - sigmoid_mod.min())


def plot_single_prediction(prediction,
                           window_size,
                           scaling_factors_dict,
                           mean_cycle_life,
                           height=600,
                           width=1000,
                           inline=False):
    """Visualizes a single model prediction and gives context by comparing it to the mean_cylce_life.
    A synthetic capacity curve is only created for better interpretability and is not based on data!
    
    Arguments:
        prediction {numpy.ndarray} -- Model output like [0.134567, 0.456787]
        
        window_size {int} -- window_size of the model
        
        scaling_factors_dict {dict} -- Feature scaling factors used during training of the model.
            This is used to make the model output interpretable as a cycle number.
            
        mean_cycle_life {int} -- Cycle life to compare the model prediction to.
    """
    relative_results = prediction / prediction.sum()  # Relative to this cell
    results_cycle_life = np.sum(np.round(
        prediction * scaling_factors_dict[REMAINING_CYCLES_NAME]).astype(np.int))
    relative_window_size = float(window_size / results_cycle_life)
    
    x_space = (0, 11, 1000)
    x_value = np.linspace(*x_space) 
    y = skewed_normalized_sigmoid(x_value)
    y_mean = skewed_normalized_sigmoid(x_value) * mean_cycle_life / results_cycle_life
    
    # Find the window indexes based on the relative size compared to results_cycle_life
    window_last_idx = (np.abs(y - relative_results[0])).argmin()
    window_first_idx = (np.abs(y - relative_results[0] + relative_window_size)).argmin()
    
    # Build a mask for the window
    window_mask = np.zeros_like(y, dtype=bool)
    window_mask[window_first_idx: window_last_idx + 1] = True
    
    x_plot = np.linspace(100, 80, x_space[-1])
    highlight_color = 'rgba(255, 128, 0, 1.0)'

    sigmoid_trace = go.Scatter(dict(
        x=y, 
        y=x_plot, 
        mode='lines',
        name='Capacity rendering',
        line=dict(color='rgba(51, 204, 204, 1.0)', width=3),
        hoverinfo='skip'))
    window_trace = go.Scatter(dict(
        x=y[window_mask], 
        y=x_plot[window_mask], 
        mode='lines',
        name='Cycle window',
        line=dict(color=highlight_color, width=3),
        showlegend=False,
        hoverinfo='skip'))
    mean_trace = go.Scatter(dict(
        x=y_mean, 
        y=x_plot, 
        mode='lines',
        line=dict(color='rgba(51, 204, 204, 0.2)', width=3),
        name='Average cycle'),
        hoverinfo='skip',
        xaxis='x2')
    dot_trace = go.Scatter(dict(
        x=[relative_results[0]], 
        y=[x_plot[window_last_idx]], 
        mode='markers',
        name='You are here!',
        marker=dict(color=highlight_color,
                    size=20),
        hoverinfo='skip'))
    
    xaxis_config = dict(
        tickmode='array',
        tickcolor='rgb(250, 250, 250)',
        zeroline=False,
        range=[-0.01, max(y.max(), y_mean.max()) + 0.03],
        showgrid=False,
    )
    
    layout = go.Layout(
        height=height,
        # width=width,
        plot_bgcolor='rgb(250, 250, 250)',
        legend=dict(
            x=.82,
            y=1,
            traceorder='normal',
            font=dict(
                family='sans-serif',
                size=12,
                color='#000'
            ),
            bgcolor='rgb(250, 250, 250)',
        ),
        xaxis=dict(
            **xaxis_config,
            overlaying='x2',  # Changes actual plotting order
            tickvals=[relative_results[0], 1],
            ticktext=[int(prediction[0] * scaling_factors_dict[REMAINING_CYCLES_NAME]),
                      str(results_cycle_life)],
            ticklen=2,
            title="Number of cycles",
            titlefont=dict(family='Arial', size=24),
            tickfont=dict(family='Arial', size=22, color=highlight_color)
        ),
        xaxis2=dict(
            **xaxis_config,
            tickvals=[y_mean.max()],
            ticktext=[str(mean_cycle_life)],
            ticklen=5,
            tickfont=dict(family='Arial', size=18, color='rgba(15, 15, 15, 0.3)')
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[80, 100],
            ticktext=["80%", "100%"],
            dtick=20,
            ticklen=5,
            tickcolor='rgb(255, 255, 255)',
            range=[80, 100],
            showgrid=False,
            title="Relative capacity",
            titlefont=dict(family='Arial', size=24),
            tickfont=dict(family='Arial', size=18)
        ), 
        shapes=[
            # Window rectangle
            {
                'type': 'rect',
                'x0': y[window_first_idx],
                'y0': 0,
                'x1': y[window_last_idx],
                'y1': x_plot[int((window_last_idx + window_first_idx) / 2)],
                'fillcolor': 'rgba(150, 150, 150, 0.1)',
                'line': {'width': 0},
                'layer': 'below'
            },
            # Line Vertical
            {
                'type': 'line',
                'x0': relative_results[0],
                'y0': 80,
                'x1': relative_results[0],
                'y1': x_plot[window_last_idx],
                'line': {'color': 'rgb(130, 130, 130)',
                         'width': 1,
                         'dash': 'dot'},
                'layer': 'below'
            }],
        annotations=[dict(
            x = max(y.max(), y_mean.max())/2,
            y=96,
            text='Cycles remaining:  {}'.format(int(prediction[1] * scaling_factors_dict[REMAINING_CYCLES_NAME])),
            showarrow=False,
            font=dict(family='Arial', size=22, color=highlight_color)
        )]
    )
    return go.Figure(data=[mean_trace, sigmoid_trace, window_trace, dot_trace], layout=layout)
