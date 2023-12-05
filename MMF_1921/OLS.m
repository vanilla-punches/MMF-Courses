function  [mu, Q, Adjusted_R2] = OLS(returns, factRet, lambda, K)
    
    % Use this function to perform an OLS regression. Note that you will 
    % not use lambda or K in thismodel (lambda is for LASSO, and K is for
    % BSS).
 
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    % T(Time): 1,...,T
    T = size(returns, 1);
    % Returns [r_1 r_2 ... r_n]: asset returns (size: T*n)
    % FactRet [f_1 f_2 ... f_p]: factor returns (size: T*p)
    % n(total number of assets)
    n = size(returns, 2);
    % p(total number of factors)
    p = size(factRet, 2);
    
    r = returns; 
    
    X = [ones(T,1) factRet]; 
    
    % Vector of regression intercepts ('alpha') and regression coefficients ('beta')
    B = ((X' * X)\ X') * r;  
    
    
    % Subtract matrix of factor loadings ('beta') only for all assets
    V = B(2:p+1,:);  
    
    % Subtract intercept ('alpha') only 
    alpha = X(1,:); 
    
    % Get the residuals resulting from the optimal regression coefficients
    epsilon = r - X * B; 
    
    % Factor covariance matrix
    F = cov(factRet);  
    
    % Matrix of residual variances with degree of freedom T-p-1
    D = (1/(T-p-1)) * vecnorm(epsilon).^2;
    
    % In-sample analysis for R^2 and adjusted R^2
    SS_tot = sum((r - mean(r,1)).^2, 1);  % sum of squares total
    SS_reg = sum((X * B - mean(r,1)).^2, 1); % sum of squares due to regression
    R_sqr = SS_reg./SS_tot;  % coefficient of determination 
    
    % Output
    mu = B' * mean(X,1)';  % n x 1 vector of asset exp. returns
    Q  = V' * F * V + diag(D);  % n x n asset covariance matrix
    
    Adjusted_R2 = 1 - ((T-1)/(T-p-1)) * (1 - R_sqr); % Adjusted R^2
    %----------------------------------------------------------------------
    
end