function R2 = computeR2(Y,Yhat)
% Compute RSquared statistics.

% get total sum of squares
SStot = sum((Y-mean(Y)).^2);

% get the residual sum of squares
SSres = sum((Y-Yhat).^2);

% compute R2
R2 = 1 - SSres/SStot;