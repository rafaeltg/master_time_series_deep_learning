
"""
    1. Forecasts
        1.1) actual x predicted (with forecast +- 2*err_std)
        1.2) metrics:
            1.2.1) Train set: CV mean and std of all metrics (rmse, r2)
            1.2.2) Test set: rmse, r2, corr
        1.3) scatter (actual x predicted)
        
    2. Errors
        2.1) box-plot of models
        2.2) scatter
        2.2) White noise (errors acf)
    
    3. Input data sets
        3.1) Stationarity test
        3.2) Train and Test set properties (len, min, max, mean, std...)
"""