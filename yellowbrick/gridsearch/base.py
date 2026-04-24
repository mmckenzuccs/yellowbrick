# yellowbrick.gridsearch.base
# Base class for grid search visualizers
#
# Author:   Phillip Schafer
# Created:  Sat Feb 3 10:18:33 2018 -0500
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: base.py [03724ed] pbs929@users.noreply.github.com $

"""
Base class for grid search visualizers
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
from ..utils import is_gridsearch
from ..base import ModelVisualizer
from ..exceptions import (
    YellowbrickTypeError,
    YellowbrickKeyError,
    YellowbrickValueError,
)


##########################################################################
## Dimension reduction utility
##########################################################################


# Function to extract the parameter values and score corresponding to 
# each gridsearch trial.
def _get_cv_result(cv_results, key, template):
    try:
        return cv_results[key]
    except KeyError:
        raise YellowbrickKeyError(template.format(key.replace("param_", "")))

# Numpy supplies a masked array, use this instead
def _map_masked_indices(values, mapping):
    mask = np.ma.getmaskarray(values)
    return [
        None if is_masked else mapping[value]
        for value, is_masked in zip(np.asarray(values), mask)
    ]

# All scores for each coordinate pair
def _accumulate_param_scores(idx_x, idx_y, scores, n_x, n_y):
    all_scores = [[None for _ in range(n_x)] for _ in range(n_y)]
    for x, y, score in zip(idx_x, idx_y, scores):
        if x is None or y is None:
            continue

        if all_scores[y][x] is None:
            all_scores[y][x] = []
        all_scores[y][x].append(score)

    return all_scores

# Get a numpy array consisting of the best scores for each parameter pair
def _best_scores_grid(all_scores, metric):
    n_y = len(all_scores)
    n_x = len(all_scores[0]) if n_y else 0
    best_scores = np.empty((n_y, n_x))

    for y in range(n_y):
        for x in range(n_x):
            if all_scores[y][x] is None:
                best_scores[y, x] = np.nan
                continue

            try:
                best_scores[y, x] = max(all_scores[y][x])
            except ValueError:
                raise YellowbrickValueError(
                    "Cannot display grid search results for metric '{}': "
                    "result values may not all be numeric".format(metric)
                )

    return best_scores


def param_projection(cv_results, x_param, y_param, metric="mean_test_score"):
    """
    Projects the grid search results onto 2 dimensions.

    The display value is taken as the max over the non-displayed dimensions.

    Parameters
    ----------
    cv_results : dict
        A dictionary of results from the `GridSearchCV` object's `cv_results_`
        attribute.

    x_param : string
        The name of the parameter to be visualized on the horizontal axis.

    y_param : string
        The name of the parameter to be visualized on the vertical axis.

    metric : string (default 'mean_test_score')
        The field from the grid search's `cv_results` that we want to display.

    Returns
    -------
    unique_x_vals : list
        The parameter values that will be used to label the x axis.

    unique_y_vals: list
        The parameter values that will be used to label the y axis.

    best_scores: 2D numpy array (n_y by n_x)
        Array of scores to be displayed for each parameter value pair.
    """
    # Extract the parameter values and score corresponding to each gridsearch
    # trial.
    # These are masked arrays where the cases where each parameter is
    # non-applicable are masked.
    x_vals = _get_cv_result(
        cv_results,
        "param_" + x_param,
        "Parameter '{}' does not exist in the grid search results",
    )
    y_vals = _get_cv_result(
        cv_results,
        "param_" + y_param,
        "Parameter '{}' does not exist in the grid search results",
    )
    scores = _get_cv_result(
        cv_results,
        metric,
        "Metric '{}' does not exist in the grid search results",
    )

    # Get unique, unmasked values of the two display parameters
    unique_x_vals = sorted(list(set(x_vals.compressed())))
    unique_y_vals = sorted(list(set(y_vals.compressed())))
    n_x = len(unique_x_vals)
    n_y = len(unique_y_vals)

    # Get mapping of each parameter value -> an integer index
    int_mapping_1 = {value: idx for idx, value in enumerate(unique_x_vals)}
    int_mapping_2 = {value: idx for idx, value in enumerate(unique_y_vals)}

    # Translate each gridsearch result to indices on the grid
    idx_x = _map_masked_indices(x_vals, int_mapping_1)
    idx_y = _map_masked_indices(y_vals, int_mapping_2)

    # Create an array of all scores for each value of the display parameters.
    # This is a n_x by n_y array of lists with `None` in place of empties
    # (my kingdom for a dataframe...)
    all_scores = _accumulate_param_scores(idx_x, idx_y, scores, n_x, n_y)
    best_scores = _best_scores_grid(all_scores, metric)

    return unique_x_vals, unique_y_vals, best_scores


##########################################################################
## Base Grid Search Visualizer
##########################################################################


class GridSearchVisualizer(ModelVisualizer):
    def __init__(self, estimator, ax=None, **kwargs):
        """
        Check to see if model is an instance of GridSearchCV.
        Should return an error if it isn't.
        """
        # A bit of type checking
        if not is_gridsearch(estimator):
            raise YellowbrickTypeError("This estimator is not a GridSearchCV instance")

        # Initialize the super method.
        super(GridSearchVisualizer, self).__init__(estimator, ax=ax, **kwargs)

    def param_projection(self, x_param, y_param, metric):
        """
        Projects the grid search results onto 2 dimensions.

        The wrapped GridSearch object is assumed to be fit already.
        The display value is taken as the max over the non-displayed dimensions.

        Parameters
        ----------
        x_param : string
            The name of the parameter to be visualized on the horizontal axis.

        y_param : string
            The name of the parameter to be visualized on the vertical axis.

        metric : string (default 'mean_test_score')
            The field from the grid search's `cv_results` that we want to display.

        Returns
        -------
        unique_x_vals : list
            The parameter values that will be used to label the x axis.

        unique_y_vals: list
            The parameter values that will be used to label the y axis.

        best_scores: 2D numpy array (n_y by n_x)
            Array of scores to be displayed for each parameter value pair.
        """
        return param_projection(self.estimator.cv_results_, x_param, y_param, metric)

    def fit(self, X, y=None, **kwargs):
        """
        Fits the wrapped grid search and calls draw().

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            Keyword arguments passed to the drawing functionality or to the
            Scikit-Learn API. See visualizer specific details for how to use
            the kwargs to modify the visualization or fitting process.

        Returns
        -------
        self : visualizer
            The fit method must always return self to support pipelines.
        """
        self.estimator.fit(X, y)
        self.draw()
        return self
