function  [mu, Q, Adjusted_R2] = LASSO(returns, factRet, lambda, K)
    
    % Use this function for the LASSO model. Note that you will not use K 
    % in this model (K is for BSS).
    %
    % You should use an optimizer to solve this problem. Be sure to comment 
    % on your code to (briefly) explain your procedure.
    
    % *************** WRITE YOUR CODE HERE ***************
    %----------------------------------------------------------------------
    
    % T: over time t = 1,...,T
    T = size(returns,1);
    % N: total # of assets
    N = size(returns,2);
    % P: total # of factors
    P = size(factRet,2);
    P_1 = P+1;
    
    returnEst=[];
    for i = 1:N
        for j = 1:size(lambda, 2)
            lamda2 = lambda(j);
            y = returns(:, i);
            % change LASSO formulation to quadprog format
            Lasso_to_Quadprog = [[ones(T, 1) factRet] zeros(T, P_1)];
            % set up Q, A and b for quadprog function
            Q = 2 * (Lasso_to_Quadprog') * Lasso_to_Quadprog;
            A = [eye(P_1)  -eye(P_1);
                 -eye(P_1) -eye(P_1)]; 
            b = zeros(2 * P_1, 1);
            
            % create auxiliary variable for each factor
            auxillaryVar= [zeros(P_1, 1) ; lamda2 * ones(P_1, 1)]';
            % create objective function
            c = ((-2 * y' * Lasso_to_Quadprog) + auxillaryVar)' ;
            % call quadprog function
            x = quadprog(Q, c, A, b, [], [], [], [], []);
            x = round(x(1:P_1), 6);
            % if constraints satisfy, the loop stops, else, the loop
            % continues to test for other lambda
            if (nnz(x) <= 5 && nnz(x) >= 2)
                returnEst = [returnEst x];
                break
            end
        end
    end 
    y = returns;
    V = returnEst(2:P_1, :);
    
    % calculate epsilon 
    X = [ones(T, 1) factRet];
    epsilon = returns - (X * returnEst);
    
    df_V = zeros(N, 1);
    for i = 1:N
        df_V(i) = nnz(V(:, i));
    end
    
    D = ones(N, 1)./(T * ones(N, 1) - df_V - ones(N, 1)) .* (vecnorm(epsilon).^2)';
    F = cov(factRet);
    
    % calculate R squared
    SS_total = sum((y - mean(y, 1)).^2, 1);
    SS_residual = sum((y - X * returnEst).^2, 1);
    R_2 = 1 - SS_residual./SS_total;
    % find adjusted R squared
    mu =   returnEst' * mean(X, 1)';       % n x 1 vector of asset exp. returns
    Q  =   V' * F * V + diag(D);       % n x n asset covariance matrix
    Adjusted_R2 = 1 - (1 - R_2) * (T - 1) ./ (T * ones(N, 1) - df_V - ones(N, 1))';    
    %----------------------------------------------------------------------
    
end