%% MMF1921 (Summer 2023) - Project 1
% 
% The purpose of this program is to implement the following factor models
% a) Multi-factor OLS regression
% b) Fama-French 3-factor model
% c) LASSO
% d) Best Subset Selection
% 
% and to use these factor models to estimate the asset expected returns and 
% covariance matrix. These parameters will then be used to test the 
% out-of-sample performance using MVO to construct optimal portfolios.
% 
% Use can use this template to write your program.
%
% Student Name:
% Student ID:

clc
clear all
format short

% Program Start
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Read input files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the stock weekly prices
adjClose = readtable('MMF1921_AssetPrices.csv');
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Date));
adjClose.Properties.RowNames = cellstr(datetime(adjClose.Properties.RowNames));
adjClose.Date = [];

% Load the factors weekly returns
factorRet = readtable('MMF1921_FactorReturns.csv');
factorRet.Properties.RowNames = cellstr(datetime(factorRet.Date));
factorRet.Properties.RowNames = cellstr(datetime(factorRet.Properties.RowNames));
factorRet.Date = [];

riskFree = factorRet(:,9);
factorRet = factorRet(:,1:8);

% Identify the tickers and the dates 
tickers = adjClose.Properties.VariableNames';
dates   = datetime(factorRet.Properties.RowNames);

% Calculate the stocks' weekly EXCESS returns
prices  = table2array(adjClose);
returns = ( prices(2:end,:) - prices(1:end-1,:) ) ./ prices(1:end-1,:);
returns = returns - ( diag( table2array(riskFree) ) * ones( size(returns) ) );
returns = array2table(returns);
returns.Properties.VariableNames = tickers;
returns.Properties.RowNames = cellstr(datetime(factorRet.Properties.RowNames));

% Align the price table to the asset and factor returns tables by
% discarding the first observation.
adjClose = adjClose(2:end,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Define your initial parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial budget to invest ($100,000)
initialVal = 100000;

% Start of in-sample calibration period 
calStart = datetime('2008-01-01');
calEnd   = calStart + calyears(4) - days(1);

% Start of out-of-sample test period 
testStart = datetime('2012-01-01');
testEnd   = testStart + calyears(1) - days(1);

% Number of investment periods (each investment period is 1 year long)
NoPeriods = 5;

% Factor models
% Note: You must populate the functios OLS.m, FF.m, LASSO.m and BSS.m with your
% own code.
FMList = {'OLS' 'FF' 'LASSO' 'BSS'}; %Original
% FMList = {'OLS' 'FF' 'LASSO'};
FMList = cellfun(@str2func, FMList, 'UniformOutput', false);
NoModels = length(FMList);

% Tags for the portfolios under the different factor models
tags = {'OLS portfolio' 'Fama-French portfolio' 'LASSO portfolio' 'BSS portfolio'};
% tags = {'OLS portfolio' 'Fama-French portfolio' 'LASSO portfolio'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Construct and rebalance your portfolios
%
% Here you will estimate your input parameters (exp. returns and cov. 
% matrix etc) from the Fama-French factor models. You will have to 
% re-estimate your parameters at the start of each rebalance period, and 
% then re-optimize and rebalance your portfolios accordingly. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initiate counter for the number of observations per investment period
toDay = 0;

% Preallocate the space for the per period value of the portfolios 
currentVal = zeros(NoPeriods, NoModels);

%--------------------------------------------------------------------------
% Set the value of lambda and K for the LASSO and BSS models, respectively
%--------------------------------------------------------------------------
lambda = 0.001:0.001:1;
K      = 4;

for t = 1 : NoPeriods
  
    % Subset the returns and factor returns corresponding to the current
    % calibration period.
    periodReturns = table2array( returns( calStart <= dates & dates <= calEnd, :) );
    periodFactRet = table2array( factorRet( calStart <= dates & dates <= calEnd, :) );
    currentPrices = table2array( adjClose( ( calEnd - days(7) ) <= dates ... 
                                                    & dates <= calEnd, :) )';
    
    % Subset the prices corresponding to the current out-of-sample test 
    % period.
    periodPrices = table2array( adjClose( testStart <= dates & dates <= testEnd,:) );
    
    % Set the initial value of the portfolio or update the portfolio value
    if t == 1
        
        currentVal(t,:) = initialVal;
        
    else
        for i = 1 : NoModels
            
            currentVal(t,i) = currentPrices' * NoShares{i};
            
        end
    end
    
    % Update counter for the number of observations per investment period
    fromDay = toDay + 1;
    toDay   = toDay + size(periodPrices,1);
    
    % Calculate 'mu' and 'Q' using the 4 factor models.
    % Note: You need to write the code for the 4 factor model functions. 
    for i = 1 : NoModels
        
        [mu{i}, Q{i}, Adjusted_R2{i}] = FMList{i}(periodReturns, periodFactRet, lambda, K);
        
    end
            
    % Optimize your portfolios to get the weights 'x'
    % Note: You need to write the code for MVO with no short sales
    for i = 1 : NoModels
        
        % Define the target return as the geometric mean of the market 
        % factor for the current calibration period
        targetRet = geomean(periodFactRet(:,1) + 1) - 1;
        
        x{i}(:,t) = MVO(mu{i}, Q{i}, targetRet); 
            
    end
    
    % Calculate the optimal number of shares of each stock you should hold
    for i = 1 : NoModels
        
        % Number of shares your portfolio holds per stock
        NoShares{i} = x{i}(:,t) .* currentVal(t,i) ./ currentPrices;
        
        % Weekly portfolio value during the out-of-sample window
        portfValue(fromDay:toDay,i) = periodPrices * NoShares{i};
        
    end

    % Update your calibration and out-of-sample test periods
    Adj_R2_list{t} = Adjusted_R2; % Updated
    calStart = calStart + calyears(1);
    calEnd   = calStart + calyears(4) - days(1);
    
    testStart = testStart + calyears(1);
    testEnd   = testStart + calyears(1) - days(1);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% 4.1 Evaluate any measures of fit of the regression models to assess their
% in-sample quality. You may want to modify Section 3 of this program to
% calculate the quality of fit each time the models are recalibrated.
%--------------------------------------------------------------------------

for t = 1:NoPeriods
    R_bar_different_model_mean{t} = [mean(Adj_R2_list{t}{1}) mean(Adj_R2_list{t}{2})...
                                mean(Adj_R2_list{t}{3}) mean(Adj_R2_list{t}{4})];
end

for t = 1:NoPeriods
    R_bar_different_model_std{t} = [std(Adj_R2_list{t}{1}) std(Adj_R2_list{t}{2})...
                                std(Adj_R2_list{t}{3}) std(Adj_R2_list{t}{4})];
end


% for t = 1:NoPeriods
%     R_bar_different_model_mean{t} = [mean(Adj_R2_list{t}{1}) mean(Adj_R2_list{t}{2})...
%                                 mean(Adj_R2_list{t}{3})];
% end
% 
% for t = 1:NoPeriods
%     R_bar_different_model_std{t} = [std(Adj_R2_list{t}{1}) std(Adj_R2_list{t}{2})...
%                                 std(Adj_R2_list{t}{3})];
% end
%--------------------------------------------------------------------------
% 4.2 Calculate the portfolio average return, variance (or standard 
% deviation), and any other performance and/or risk metric you wish to 
% include in your report.
%--------------------------------------------------------------------------

% Portfolio Value
portfValue = [100000*ones(1,size(portfValue,2)); portfValue];

% Portfolio Returns
portf_Return = (portfValue(2:end,:)-portfValue(1:end-1,:))./portfValue(1:end-1,:);

% Portfolio average return
portf_Return_mean = mean(portf_Return,1);

% Portfolio standard deviation
portf_Return_std = std(portf_Return,1);

% Portfolio Sharp ratio
portf_sharpRatio = (portf_Return_mean-mean(table2array(riskFree)))./portf_Return_std;

%--------------------------------------------------------------------------
% 4.3 Plot the portfolio wealth evolution 
% 
% Note: The code below plots all portfolios onto a single plot. However,
% you may want to split this into multiple plots for clarity, or to
% compare a subset of the portfolios. 
%--------------------------------------------------------------------------
plotDates = dates(dates >= datetime('2012-01-01') );

fig1 = figure(1);

for i = 1 : NoModels
    plot( plotDates, portfValue(2:end,i) )
    hold on
end

legend(tags, 'Location', 'eastoutside','FontSize',12);
datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
set(gca,'XTickLabelRotation',30);
title('Portfolio value', 'FontSize', 14)
ylabel('Value','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(fig1,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(fig1,'Position');
set(fig1,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(fig1,'fileName','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(fig1,'fileName','-dpng','-r0');

%--------------------------------------------------------------------------
% 4.4 Plot the portfolio weights period-over-period
%--------------------------------------------------------------------------

% OLS portfolio weights
figure_OLS = figure(2);
area(x{1}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('OLS portfolio weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Fama-French portfolio weights
figure_FF = figure(3);
area(x{2}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('Fama-French portfolio weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% LASSO portfolio weights
figure_LASSO = figure(4);
area(x{3}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('LASSO portfolio weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% BSS portfolio weights
figure_BSS = figure(5);
area(x{4}')
legend(tickers, 'Location', 'eastoutside','FontSize',12);
title('BSS portfolio weights', 'FontSize', 14)
ylabel('Weights','interpreter','latex','FontSize',12);
xlabel('Rebalance period','interpreter','latex','FontSize',12);

% Define the plot size in inches
set(figure_OLS,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(figure_OLS,'Position');
set(figure_OLS,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

set(figure_FF,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(figure_FF,'Position');
set(figure_FF,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

set(figure_LASSO,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(figure_LASSO,'Position');
set(figure_LASSO,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

set(figure_BSS,'Units','Inches', 'Position', [0 0 8, 5]);
pos1 = get(figure_BSS,'Position');
set(figure_BSS,'PaperPositionMode','Auto','PaperUnits','Inches',...
    'PaperSize',[pos1(3), pos1(4)]);

% If you want to save the figure as .pdf for use in LaTeX
% print(figure_OLS,'fileName2','-dpdf','-r0');

% If you want to save the figure as .png for use in MS Word
print(figure_OLS,'OLS_weight','-dpng','-r0');
print(figure_FF,'Fama-French_weight','-dpng','-r0');
print(figure_LASSO,'LASSO_weight','-dpng','-r0');
print(figure_BSS,'BSS_weight','-dpng','-r0');

%--------------------------------------------------------------------------
% 4.5 Different K Value
%--------------------------------------------------------------------------
% %% Initial parameters
% 
% % Start of in-sample calibration period 
% calStart = datetime('2008-01-01');
% calEnd   = calStart + calyears(4) - days(1);
% 
% % Start of out-of-sample test period 
% testStart = datetime('2012-01-01');
% testEnd   = testStart + calyears(1) - days(1);
% 
% K = [3 4 5 6 7];
% tagsK = {'K = 3' 'K = 4' 'K = 5' 'K = 6' 'K = 7'};
% NoK = length(tagsK);

% %% Construct and rebalance portfolios
% toDay = 0;
% currentValK = zeros(NoPeriods, NoK);
% 
% for t = 1 : NoPeriods
%     periodReturnsK = table2array( returns  ( calStart <= dates & dates <= calEnd, :) );
%     periodFactRetK = table2array( factorRet( calStart <= dates & dates <= calEnd, :) );
%     currentPricesK = table2array( adjClose ( ( calEnd - days(7) ) <= dates & dates <= calEnd, :) )';
%     periodPricesK  = table2array( adjClose ( testStart <= dates & dates <= testEnd,:) );
% 
%     if t == 1
%         currentValK(t,:) = initialVal;  
%     else
%         for i = 1 : NoK
%             currentValK(t,i) = currentPricesK' * NoSharesK{i}; 
%         end
%     end
% 
%     fromDay = toDay + 1;
%     toDay   = toDay + size(periodPricesK,1);
% 
%     for i = 1 : NoK
%         [muK{i}, QK{i}, R_square_adjK{i}] = BSS(periodReturnsK, periodFactRetK, lambda, K(i)); 
%     end
% 
%     for i = 1 : NoK
%         targetRetK = geomean(periodFactRetK(:,1) + 1) - 1;
%         xK{i}(:,t) = MVO(muK{i}, QK{i}, targetRetK);       
%     end
% 
%     for i = 1 : NoK
%         NoSharesK{i} = xK{i}(:,t) .* currentValK(t,i) ./ currentPricesK;
%         portfValueK(fromDay:toDay,i) = periodPricesK * NoSharesK{i};   
%     end
% 
%     R_square_list{t} = R_square_adjK;
% 
%     calStart = calStart + calyears(1);
%     calEnd   = calStart + calyears(4) - days(1);
% 
%     testStart = testStart + calyears(1);
%     testEnd   = testStart + calyears(1) - days(1);
% 
% end
% 
% % In-sample analysis of different K
% for t = 1:NoPeriods
%     R2_mean_BSS{t} = [mean(R_square_list{t}{1}) mean(R_square_list{t}{2})...
%                       mean(R_square_list{t}{3}) mean(R_square_list{t}{4})...
%                       mean(R_square_list{t}{5})];
% end
% 
% for t = 1 : NoPeriods
%     R2_std_BSS{t} = [std(R_square_list{t}{1}) std(R_square_list{t}{2})...
%                      std(R_square_list{t}{3}) std(R_square_list{t}{4})...
%                      std(R_square_list{t}{5})];
% end
% % In-sample analysis of different K
% 
% % Compute the annulized average excess return and volatility
% 
% % Annulized portfolio returns 
% portfValueK = [100000*ones(1,size(portfValueK,2));portfValueK];
% 
% % Portfolio Returns
% portf_ReturnK = (portfValueK(2:end,:)-portfValueK(1:end-1,:))./portfValueK(1:end-1,:);
% 
% % Portfolio average return
% portf_ReturnK_mean = mean(portf_ReturnK,1);
% 
% % Portfolio standard deviation
% portf_ReturnK_std = std(portf_ReturnK,1);
% 
% % Portfolio Sharp ratio
% portfK_sharpRatio = (portf_ReturnK_mean-mean(table2array(riskFree)))./portf_ReturnK_std;
% 
% plotDates = dates(dates >= datetime('2012-01-01') );
% 
% fig6 = figure(6);
% 
% for i = 1 : NoK
%     plot(plotDates, portfValueK(2:end,i), 'LineWidth', 1)
%     hold on
% end
% 
% legend(tagsK, 'Location', 'eastoutside','FontSize',12);
% datetick('x','dd-mmm-yyyy','keepticks','keeplimits');
% set(gca,'XTickLabelRotation',30);
% title('Portfolio value of different K', 'FontSize', 14)
% ylabel('Value','interpreter','latex','FontSize',12);
% 
% % Define the plot size in inches
% set(fig6,'Units','Inches', 'Position', [0 0 8, 5]);
% pos1 = get(fig6,'Position');
% set(fig6,'PaperPositionMode','Auto','PaperUnits','Inches',...
%     'PaperSize',[pos1(3), pos1(4)]);
% 
% % If you want to save the figure as .pdf for use in LaTeX
% % print(fig1,'fileName','-dpdf','-r0');
% 
% % If you want to save the figure as .png for use in MS Word
% print(fig6,'Portfolio Value K','-dpng','-r0');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Program End