from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np
import pandas as pd

from trainer import constants as cst

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
        
    results_df = pd.DataFrame({
        "pred_current_cycle": predictions[:, 0],
        "pred_remaining_cycles": predictions[:, 1],
        "target_current_cycle": targets[:, 0],
        "target_remaining_cycles": targets[:, 1],
    })
    
    return results_df


def create_cell_index(results_df, cell_index_col_name="cell_index", inplace=False):
    """Takes a results datafram from get_predictions_results and adds a new column
    with an integer index for every entry which belongs to the same cell.
    
    The indexes do not correspond to the actual indexes in the original data!
    """
    # Initialization
    if inplace:
        results = results_df
    else:
        results = results_df.copy()
    results[cell_index_col_name] = 0
    
    # Getting the border indexes for all cells
    new_cell_index = list(results[results["target_current_cycle"].diff() < 0].index)
    new_cell_index.append(len(results))  # Add the last index manually, since there is no diff < 0
    last_s = 0  # Set first starting index manually
    
    # Setting cell_indexes
    for i, s in enumerate(new_cell_index):
        results[cell_index_col_name].iloc[last_s:s] = i
        last_s = s
    
    if not inplace:
        return results


def plot_predictions_and_errors(results_df, height=1300, width=4000, return_div=True):
    """Plots predictions vs. target and the corresponding absolute errors
    for current and remaining cycles.
    
    if return_div == False, a normal plotly plot is created and opended in a new tab.
    Otherwise the returned <div> element may be used for wrapping the plot in html. 
    """
    
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
        height=height,
        width=width,
        yaxis=dict(domain=[0.7, 1]),
        yaxis2=dict(domain=[0.5, 0.7]),
        yaxis3=dict(domain=[0.2, 0.5]),
        yaxis4=dict(domain=[0, 0.2])
    )
    
    if return_div:
        div = pyo.plot(fig, output_type='div')
        return div
    else:
        pyo.plot(fig)


def get_binned_cycle_count_trace(results_df, window_size, cycle_bin_width=100, column="target_current_cycle"):
    """Creates a plotly line trace with the value counts of results_df[column] which is used
    in plot_errors_and_counts.
    
    window_size is needed for correct binning of the valuecounts."""
    
    # Get the current cycle value counts sorted from low to high
    current_cycle_counts = (results_df[column]
                            .value_counts()
                            .sort_index()
                            .reset_index()
                            .rename(columns={"index": "current_cycle_value",
                                             column: "count"}))

    # The actual cycle counts can be {20: 42,  21: 2,  25: 42,  26: 2} since some cycles were dropped
    # Binning aggregates these "outliers" with bin size equal to shift
    bins = list(range(window_size, results_df[column].max(), cycle_bin_width))
    bins.append(results_df[column].max())
    grouped_cycle_counts = (current_cycle_counts
                            .groupby(pd.cut(current_cycle_counts["current_cycle_value"], bins=bins))
                            .sum()
                            .loc[:, "count"])
    
    # Convert to percent, since the absolute counts can vary widely, when cycle_bin_width changes
    grouped_cycle_counts = ((grouped_cycle_counts - grouped_cycle_counts.min())
                            / (grouped_cycle_counts.max() - grouped_cycle_counts.min()))

    return go.Scatter(x=np.array(bins) - window_size,  # shift necessary to line up with error traces
                      y=grouped_cycle_counts,
                      name="Cells count")


def get_errors_over_cycle_traces(results_df, cycle_bin_width=100):
    """Creates a plotly bar trace with the mean absolute errors for current and remaining cycles
    aggregated in bins of "target_current_cycle".
    This shows the different levels of errors of the model during different cycle ranges.  
    
    This trace is used in plot_errors_and_counts.
    """
    results = results_df.copy()

    # Calculate absolute errors
    results["ae_current_cycle"] = (results["target_current_cycle"] - results["pred_current_cycle"]).abs()
    results["ae_remaining_cycles"] = (results["target_remaining_cycles"] - results["pred_remaining_cycles"]).abs()
    
    # Create bin intervalls
    bins = list(range(0, results["target_current_cycle"].max(), cycle_bin_width))
    bins.append(results["target_current_cycle"].max())
    
    # Aggregate mean absolute errors over bins and save as new dataframe
    mae_binned = (results.groupby(pd.cut(results_df["target_current_cycle"], bins=bins))
                  .mean()
                  .loc[:, ["ae_current_cycle", "ae_remaining_cycles"]])
    
    std_binned = (results.groupby(pd.cut(results_df["target_current_cycle"], bins=bins))
                  .std()
                  .loc[:, ["ae_current_cycle", "ae_remaining_cycles"]])
    
    # Build mean absolute errors over bins
    mae_current_cycle_trace = go.Bar(
        x=bins,
        y=mae_binned["ae_current_cycle"],
        name="mae_current_cycle"
    )
    mae_remaining_cycles_trace = go.Bar(
        x=bins,
        y=mae_binned["ae_remaining_cycles"],
        name="mae_remaining_cycles"
    )
    
    # Build standard deviation of absolute errors over bins
    std_current_cycle_trace = go.Bar(
        x=bins,
        y=std_binned["ae_current_cycle"],
        name="std_current_cycle"
    )
    std_remaining_cycles_trace = go.Bar(
        x=bins,
        y=std_binned["ae_remaining_cycles"],
        name="std_remaining_cycles"
    )

    return (mae_current_cycle_trace,
            mae_remaining_cycles_trace,
            std_current_cycle_trace,
            std_remaining_cycles_trace)


def plot_errors_and_counts(results_df,
                           window_size,
                           height=800,
                           width=1000,
                           cycle_bin_width=100,
                           show_count=True,
                           inline=False):
    """Plots the traces from get_errors_over_cycle_traces and get_binned_cycle_count_trace side by side.
    If show_count == False, then only the errors will be plotted in a single graph (height and width stay the same).
    """
    mae_cc, mae_rc, _, _ = get_errors_over_cycle_traces(results_df, cycle_bin_width)
        
    if show_count:
        count_trace = get_binned_cycle_count_trace(results_df, window_size, cycle_bin_width)
        count_trace.update(dict(
            # mode= 'none',
            line=dict(color="rgba(210, 210, 210, 1.0)"),
            fill='tozeroy',
            fillcolor="rgba(210, 210, 210, 0.5)",
            yaxis="y2",
        ))
    
    # # (WIP) If ticks for errors and cell percentage should line up.
    # dtick_error = 100
    # max_error = max(mae_cc.y.max(), mae_rc.y.max())
    # tickvals_error = list(range(0, int(max_error), dtick_error))
    # print(max(tickvals_error) / max_error)

    layout = dict(
        height=height,
        width=width,
        xaxis=dict(
            title="Cycle",
            titlefont=dict(family='Arial', size=24),
            tickfont=dict(family='Arial', size=18)
        ),
        yaxis=dict(
            title="Mean absolute error",
            # overlaying="y2",
            # tickmode="array",
            # tickvals=tickvals_error,
            dtick=200,
            titlefont=dict(family='Arial', size=24),
            tickfont=dict(family='Arial', size=18),
        ),
        yaxis2=dict(
            title="Cell count",
            tickformat='%',
            # side="right",
            tickmode="array",
            titlefont=dict(family='Arial', size=24),
            tickfont=dict(family='Arial', size=18),
        )
    )
    
    if show_count:
        fig = tools.make_subplots(rows=2, shared_xaxes=True)
        fig.append_trace(mae_cc, 1, 1)
        fig.append_trace(mae_rc, 1, 1)
        fig.append_trace(count_trace, 2, 1)
        layout["yaxis"].update(dict(domain=[0.4, 1.0]))
        layout["yaxis2"].update(dict(domain=[0.0, 0.3]))
        fig["layout"].update(layout)
    else:
        fig = go.Figure(data=[mae_cc, mae_rc], layout=layout)

    if inline:
        pyo.iplot(fig)
    else:
        pyo.plot(fig)