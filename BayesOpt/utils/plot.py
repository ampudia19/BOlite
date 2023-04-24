import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False
    })

######## Figure 2.2
def fig22(xtrain, xtest, y1, y3, vprior, ypost, ymean, vmean):
    """
    Plots Figure 2.2 from R&W 2006. Uses train and test data, single and multi-trajectories
    for both the prior and posterior distributions to build Figure.
    """
    fig, ax = plt.subplots(1,2, figsize=(16,4))
    # Figure 2.2a
    for trajectory in range(y3.shape[1]):
        ax[0].plot(xtest, y3[:, trajectory])
        ax[0].fill_between(
            xtest, 
            0 - 2 * vprior**0.5, 
            0 + 2 * vprior**0.5,
            color="gray", 
            alpha=0.15
        )
    # Figure 2.2b
    for trajectory in range(ypost.shape[1]):
        ax[1].plot(xtest, ypost[:, trajectory])
    ax[1].fill_between(
        xtest, 
        np.mean(ymean, axis=1) - 2 * vmean**0.5, 
        np.mean(ymean, axis=1) + 2 * vmean**0.5,
        color="gray", 
        alpha=0.3
    )
    for axis in ax:
        axis.set_ylabel("output, $f(x)$")
        axis.set_xlabel("input, $x$")
        axis.set_ylim(-3.2, 3.2)
        axis.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax[1].scatter(xtrain, y1, marker="^", s=80, c="r")
    plt.show()

######## Figure 2.5
def fig25(xtrain, y1, xtest, y25a, y25b, y25c, v25a, v25b, v25c, suptitle="fig25abc"):
    """
    Plots Figure 2.5 from R&W 2006. Uses train and test data, and multi-trajectories
    with different kernel parameters to build the Figure.
    """
    gs = matplotlib.gridspec.GridSpec(5, 16)
    ax0 = plt.subplot(gs[:2, 4:12])
    ax1 = plt.subplot(gs[3:5, 9:])
    ax2 = plt.subplot(gs[3:5, :7])
    subfig25(ax0, xtrain, y1, xtest, y25a, v25a, "$lengthscale=1$")
    subfig25(ax1, xtrain, y1, xtest, y25c, v25c, "$lengthscale=3$")
    subfig25(ax2, xtrain, y1, xtest, y25b, v25b, "$lengthscale=0.3$")
    plt.show()

def subfig25(axis, xtrain, y1, xtest, y, v, title):
    """
    Subfigure in Figure 2.5.
    """
    axis.plot(xtest, np.squeeze(y))
    axis.fill_between(
        xtest, 
        np.squeeze(y) - 2 * v**0.5, 
        np.squeeze(y) + 2 * v**0.5,
        color="gray", 
        alpha=0.15
    )
    axis.scatter(xtrain, y1, marker="+", s=80, c="b", alpha=0.6)
    axis.set_ylabel("output, $f(x)$")
    axis.set_xlabel("input, $x$")
    axis.set_ylim(-3.5, 3.5)
    axis.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axis.set_title(title)

######## Function Plot
def plot_function(
    target,
    mins: np.ndarray,
    maxs: np.ndarray,
    grid: int,
    title: str,
    xlabel: str,
    ylabel: str
):
    """
    Plots the Branin function (or other 2-dimensional functions) in a 3D grid.
    """
    xspace = np.linspace(mins[0], maxs[0], grid)
    yspace = np.linspace(mins[1], maxs[1], grid)
    xx, yy = np.meshgrid(xspace, yspace)
    Xcomb = np.vstack((xx.flatten(), yy.flatten())).T
    
    output = target(*Xcomb.T)

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "surface"}]],
        subplot_titles=title,
    )

    out_df = pd.DataFrame(output.reshape([xx.shape[0], yy.shape[1]]))

    fig.add_trace(
        go.Surface(z=out_df, x=xx, y=yy, showscale=False, opacity=0.9, colorscale="geyser"),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    fig.update_yaxes(title_text=ylabel, row=1, col=1)

    return fig

def add_points(x, y, z, fig, num_init, figrow=1, figcol=1):
    """
    Adds 2D points from an instance of `class.BayesianOptimizer`.
    """
    n_points = x.shape[0]

    col_pts, mark_pts = format_point_markers(n_points, num_init)

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=4, color=col_pts, symbol=mark_pts, opacity=0.9)
        ),
        row=figrow,
        col=figcol,
    )

    return fig

def format_point_markers(
    n_points,
    num_init,
    m_init="x",
    m_add="circle"
):
    """
    Prepares point marker styles according to some BO factors. Initial points are 
    shown as X, new explored points are circles. Brighter colors correspond to
    late-draws in the optimization schedule.
    """
    col_pts = list(np.arange(n_points))
    mark_pts = np.repeat(m_init, n_points).astype("<U15")
    mark_pts[num_init:] = m_add

    return col_pts, mark_pts

def main_figure(output_dict):
    fig, ax = plt.subplots(1,3, figsize=(18,6))
    plt.suptitle("Bayesian Optimization under EI, PoI and UCB for different configurations of $\epsilon$", size=16)
    ax[0].plot(np.arange(5,105,1), output_dict["ei"][-0.5][5:], label="$\epsilon = -0.5$")
    ax[0].plot(np.arange(5,105,1), output_dict["ei"][0][5:], label="$\epsilon = 0$")
    ax[0].plot(np.arange(5,105,1), output_dict["ei"][0.5][5:], label="$\epsilon = 0.5$")
    ax[0].legend(loc="upper right", prop={'size': 12})
    ax[0].set_title("Expected Improvement", size=14)
    ax[1].plot(np.arange(5,105,1), output_dict["poi"][-0.5][5:], label="$\epsilon = -0.5$")
    ax[1].plot(np.arange(5,105,1), output_dict["poi"][0][5:], label="$\epsilon = 0$")
    ax[1].plot(np.arange(5,105,1), output_dict["poi"][0.5][5:], label="$\epsilon = 0.5$")
    ax[1].legend(loc="upper right", prop={'size': 12})
    ax[1].set_title("Probability of Improvement", size=14)
    ax[2].plot(np.arange(5,105,1), output_dict["ucb"][0.5][5:], label="$\epsilon = 0.5$")
    ax[2].plot(np.arange(5,105,1), output_dict["ucb"][1][5:], label="$\epsilon = 1$")
    ax[2].plot(np.arange(5,105,1), output_dict["ucb"][1.5][5:], label="$\epsilon = 1.5$")
    ax[2].legend(loc="upper right", prop={'size': 12})
    ax[2].set_title("Upper Confidence Bound", size=14)
    for axis in ax:
        axis.set_ylabel("Mean Squared Error")
        axis.set_xlabel("Function evaluation")
    plt.show()