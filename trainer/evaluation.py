from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np
import pandas as pd

from trainer import constants as cst
from trainer import data_pipeline as dp

# TODO Feature values


def get_predictions_results(model, dataset, scaling_factors_dict):

    predictions = []
    targets = []
    
    for i, (example, target) in enumerate(dataset):
        predictions.extend(model.predict(example).tolist())
        targets.extend(target.numpy().tolist())
    
    if scaling_factors_dict:
        # Scale to original range and round for floating point errors of conversion.
        predictions = np.round(np.array(predictions) * scaling_factors_dict[cst.REMAINING_CYCLES_NAME]).astype(np.int)
        targets = np.round(np.array(targets) * scaling_factors_dict[cst.REMAINING_CYCLES_NAME]).astype(np.int)
    else:
        predictions = np.array(predictions)
        targets = np.array(targets)
        
    results = pd.DataFrame({
        "pred_current_cycle": predictions[:, 0],
        "pred_remaining_cycles": predictions[:, 1],
        "target_current_cycle": targets[:, 0],
        "target_remaining_cycles": targets[:, 1],
    })
    
    return results


def plot_predictions_and_errors(results_df, auto_open=False):
    
    x_values = np.arange(len(results_df))
    
    # Target current cycle
    target_current_cycle_trace = go.Scatter(dict(
        x=x_values, 
        y=results_df["target_current_cycle"], 
        mode='lines+markers', 
        name='Current cycle target'
    ))
    
    # Predicted current cycle
    pred_current_cycle_trace = go.Scatter(dict(
        x=x_values, 
        y=results_df["pred_current_cycle"], 
        mode='lines+markers', 
        name='Current cycle prediction'
    ))
    
    # Absolute error current cycle
    ae_current_cycle = (results_df["pred_current_cycle"] - results_df["target_current_cycle"]).abs().values
    ae_current_cycle_trace = go.Scatter(dict(
        x=x_values, 
        y=ae_current_cycle, 
        mode='lines+markers', 
        name='Current cycle absolute error'
    ))
    
    # Target remaining cycles
    target_remaining_cycles_trace = go.Scatter(dict(
        x=x_values, 
        y=results_df["target_remaining_cycles"], 
        mode='lines+markers', 
        name='Remaining cycles target'
    ))
    
    # Predicted remaining cycles
    pred_remaining_cycles_trace = go.Scatter(dict(
        x=x_values, 
        y=results_df["pred_remaining_cycles"], 
        mode='lines+markers', 
        name='Remaining cycles prediction'
    ))
    
    # Absolute error remaining cycles
    ae_remaining_cycles = (results_df["pred_remaining_cycles"] - results_df["target_remaining_cycles"]).abs().values
    ae_remaining_cycles_trace = go.Scatter(dict(
        x=x_values, 
        y=ae_remaining_cycles, 
        mode='lines+markers', 
        name='Remaining cycles absolute error'
    ))
    
    fig = tools.make_subplots(rows=4, cols=1, shared_xaxes=True)
    
    fig.append_trace(target_current_cycle_trace, 1, 1)
    fig.append_trace(pred_current_cycle_trace, 1, 1)
    fig.append_trace(ae_current_cycle_trace, 2, 1)
    
    fig.append_trace(target_remaining_cycles_trace, 3, 1)
    fig.append_trace(pred_remaining_cycles_trace, 3, 1)
    fig.append_trace(ae_remaining_cycles_trace, 4, 1)
    
    fig['layout'].update(
        height=1300,
        width=4000,
        yaxis=dict(domain=[0.7, 1]),
        yaxis2=dict(domain=[0.5, 0.7]),
        yaxis3=dict(domain=[0.2, 0.5]),
        yaxis4=dict(domain=[0, 0.2])
    )
    
    div = pyo.plot(fig, output_type='div', auto_open=auto_open)
    return div
