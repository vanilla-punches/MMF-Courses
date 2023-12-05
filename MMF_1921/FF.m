function  [mu, Q, Adjusted_R2] = FF(returns, factRet, lambda, K)
    
    % Use this function to calibrate the Fama-French 3-factor model. Note 
    % that you will not use lambda or K in this model (lambda is for LASSO, 
    % and K is for BSS).
 
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    % T(Time): 1,...,T
    T = size(returns, 1);
    % Returns [r_1 r_2 ... r_n]: asset returns (size: T*n)
    % FactRet [f_1 f_2 ... f_p]: factor returns (size: T*p)
    % n(total number of assets)
    n = size(returns, 2);
    % p: total # of factors, in FF model, we have 3 factors.
    p = 3;
    
    r = returns; 
    
    factRet_FF = factRet(:,1:3);  % subset of factor return for FF model, only three factors
    
    % data matrix X that includes a column of ones to account for the intercept of 
    % regression (?alpha?) and each of other columns correspond to a single factor over timet=1,...,T
    X = [ones(T,1) factRet_FF];  % since we only keep three factors
    
    % vector of regression intercepts ('alpha') and regression coefficients ('beta')
    B = ((X' * X)\X') * r;  
    
    % subtract matrix of factor loadings ('beta') only for all assets
    V = B(2:p+1,:);  
    
    % subtract intercept ('alpha') only 
    alpha = X(1,:);  
    
    % get the residuals resulting from the optimal regression coefficients
    epsilon = r - X * B; 
    
    % factor covariance matrix
    F = cov(factRet_FF);  
    
    % matrix of residual variances with degree of freedom T-p-1
    % we have assumption that cov(epsilon_i, epsilon_j) = 0
    % D = diag(cov(epsilon)) * ((size(r,1)-1)/(size(r,1)-size(V,1)-1));
    D = (1/(T-p-1)) * vecnorm(epsilon).^2;

    % in-sample analysis for R^2 and adjusted R^2
    SS_tot = sum((r - mean(r,1)).^2, 1);  % sum of squares total
    SS_reg = sum((X * B - mean(r,1)).^2, 1); % sum of squares due to regression
    R_sqr = SS_reg./SS_tot;  % coefficient of determination 
    
    % output
    mu = B' * mean(X,1)';  % n x 1 vector of asset exp. returns
    Q  = V' * F * V + diag(D);  % n x n asset covariance matrix
   
    Adjusted_R2 = 1 - ((T-1)/(T-p-1)) * (1 - R_sqr); % Adjusted R^2
    %----------------------------------------------------------------------
    
end