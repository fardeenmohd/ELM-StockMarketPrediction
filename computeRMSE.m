function RMSE = computeRMSE(Y,Yhat)
% Compute Root Mean Square Error statistics.
% Author : Fardin Mohammed
% Date of Revision : 27.11.2017

% compute RMSE
RMSE = sqrt(mean((Y - Yhat).^2));