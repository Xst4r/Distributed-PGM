import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from scipy.interpolate import BSpline, make_interp_spline


def smooth_plot(data1, data2, n_samples):
    """
    Interpolation to smooth data for a nicer visualization.
    We use Bsplines to interpolate (x,y).

    Parameters
    ----------
    data1 : array
       The x data

    data2 : array
       The y data

    n_samples : integer
        Number of points to interpolate.

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    xnew = np.linspace(data1.min(), data1.max(), n_samples)
    spl = make_interp_spline(data1, data2, k=3)
    return spl(xnew)


def score_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    return out


def graph_plotter(ax, edgelist):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    g = nx.from_edgelist(edgelist)

    graph_opts = {"node_size": 300,
                  "node_color": 'w',
                  "node_shape": 'o',
                  "alpha": 1.0,
                  "cmap": None,
                  "linewidths": 1.0,
                  "width": 1.0,
                  "edge_color": 'k',
                  "edge_cmap": None,
                  "style": 'solid',
                  "labels": None,
                  "font_size": 12,
                  "font_colors": 'k',
                  "font_weight": 'normal',
                  "font_family": 'sans-serif'
                  }
    nx.draw(g, ax=ax, **graph_opts)
    return ax



data1, data2, data3, data4 = np.random.randn(4, 100)
graph = np.array([[0,1], [1,2], [2,3], [3,4]])
fig, ax = plt.subplots(1, 1)
#score_plotter(ax, data1, data2, {'marker': 'x'})
graph_plotter(ax, graph)