# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:33:50 2023

@author: BX98LW
"""


from anomaly_estimation_space_time_VAR import anomaly_estimation


if __name__ == "__main__":
    times = 1420
    pred_months = 12
    lat = 10
    lon = 10

    # =============================================================================
    # Example 1
    # =============================================================================
    pred_idx_lon, pred_idx_lat = [5], [5]
    pred_idx = [pred_idx_lon[0]*10+pred_idx_lat[0]]
    anomaly_estimation("1", times, pred_months, lat, lon, pred_idx_lon, pred_idx_lat, pred_idx)

    # =============================================================================
    # Example 2
    # =============================================================================
    pred_idx_lon, pred_idx_lat = [5,6], [4,5]
    pred_idx = [pred_idx_lon[0]*10+pred_idx_lat[0], pred_idx_lon[1]*10+pred_idx_lat[0],
                pred_idx_lon[0]*10+pred_idx_lat[1], pred_idx_lon[1]*10+pred_idx_lat[1]]
    anomaly_estimation("2", times, pred_months, lat, lon, pred_idx_lon, pred_idx_lat, pred_idx)

    # =============================================================================
    # Example 3
    # =============================================================================
    pred_idx_lon, pred_idx_lat = [4, 5, 6], [4, 5, 6]
    pred_idx = [pred_idx_lon[0]*10+pred_idx_lat[0], pred_idx_lon[1]*10+pred_idx_lat[0], pred_idx_lon[2]*10+pred_idx_lat[0],
                pred_idx_lon[0]*10+pred_idx_lat[1], pred_idx_lon[1]*10+pred_idx_lat[1], pred_idx_lon[2]*10+pred_idx_lat[1],
                pred_idx_lon[0]*10+pred_idx_lat[2], pred_idx_lon[1]*10+pred_idx_lat[2], pred_idx_lon[2]*10+pred_idx_lat[2]]
    anomaly_estimation("3", times, pred_months, lat, lon, pred_idx_lon, pred_idx_lat, pred_idx)

    

    