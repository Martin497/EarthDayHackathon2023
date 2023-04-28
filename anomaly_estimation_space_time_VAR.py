# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:14:23 2023

@author: BX98LW
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import csv


def multivariate_OLS(X, Y):
    """

    Compute OLS for a multivariate regression problem.

    Parameters
    ----------
    X : ndarray, size=(n, r)
        Design matrix.
    Y : ndarray, size=(n, k)
        Target.

    Returns
    -------
    B : ndarray, size=(r, k)
        Parameter matrix.
    eps : ndarray, size=(n, k)
        Residuals.

    """
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    eps = Y - X @ B
    return B, eps


def anomaly_estimation(ex, times, pred_months, lat, lon, pred_idx_lon, pred_idx_lat, pred_idx):
    """
    This method

    Parameters
    ----------
    ex : str
        The example number. Options are '1', '2', or '3'.
    times : int
        Number of times.
    pred_months : int
        Number of months to forecast.
    lat : int
        Latitude measurements.
    lon : int
        Longitude measurements.
    pred_idx_lon : list
        Indices for longitude forecasting.
    pred_idx_lat : list
        Indices for latitude forecasting.
    pred_idx : list
        Indices for latitude forecasting in the flattened array.
    """
    ds = nc.Dataset(f"data/temp_anomaly_ex{ex}.nc")
    print(ds)
    ta = ds["tempanomaly"][:].data

    pred_idx_in = list(lon*lat+np.array(pred_idx))

    # =============================================================================
    # Fitting
    # =============================================================================
    pred_dim = len(pred_idx_lon)*len(pred_idx_lat)
    training_in = np.zeros((times-1, 2*lat*lon-pred_dim))
    training_out = np.zeros((times-1, pred_dim))

    for t in range(times-pred_months-1):
        training_in[t, :] = np.delete(np.hstack((ta[t, :, :].flatten(), ta[t+1, :, :].flatten())), pred_idx_in)
        training_out[t, :] = ta[t+1, :, :].flatten()[pred_idx]

    pars, eps = multivariate_OLS(training_in[:-pred_months, :], training_out[:-pred_months])

    # =============================================================================
    # Forecasting
    # =============================================================================
    pred_start_time = times-pred_months-1
    pred = np.zeros((pred_months, pred_dim))
    pred_eps = np.zeros((pred_months, pred_dim))
    test_in = np.delete(np.hstack((ta[pred_start_time, :, :].flatten(), ta[pred_start_time+1, :, :].flatten())), pred_idx_in)
    for t in range(pred_months):
        if t > 0:
            test_in = np.delete(np.hstack((ta[pred_start_time+t, :, :].flatten(), ta[pred_start_time+t+1, :, :].flatten())), pred_idx_in)
            test_in[pred_idx] = pred[t-1, :]
        pred[t, :] = test_in @ pars
        pred_eps[t, :] = ta[pred_start_time+t+1, :, :].flatten()[pred_idx] - pred[t, :]

    dict_ = {}
    for idxi, i in enumerate(pred_idx_lon):
        for idxj, j in enumerate(pred_idx_lat):
            dict_[f"({i+1}, {j+1})"] = pred.reshape(-1, len(pred_idx_lon), len(pred_idx_lat))[:, idxj, idxi]

    df = pd.DataFrame(dict_)
    df.to_csv(f'VAR_pred_ex{ex}.csv')
    df["(6, 6)"].to_csv(f'VAR_pred_66_ex{ex}.csv')

    # with open(f'VAR_pred_66_ex{ex}.csv', 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     # write the data
    #     writer.writerow(df["(6, 6)"])