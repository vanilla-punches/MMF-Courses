from pickle import FALSE
import numpy as np

def CRROptionWGreeks_Barrier(currStockPrice, strikePrice, barrier, intRate, divYield, vol, totSteps, yearsToExp, optionType, american):
    """ BinomialOptionPricer implements a simple binomial model for option pricing
    
        currStockPrice = Current price of the underlying stock
        strikePrice = Strike price of the option
        intRate = Annualized continuous compounding interest rate
        divYield = Annualized continuous compounding dividend yield
        vol = Expected forward annualized stock price volatility
        totStep = Depth of the tree model
        yearsToExp = Time to expiry in years
        optionType = "CALL" or "PUT"
        american = true or false, i.e. is early excercise allowed (true = American option, false = European)
    
        returns the calculated price of the option
    """

    # calculate the number of time steps that we require
    timeStep = yearsToExp / totSteps
      
    # one step random walk (price increases)
    u = np.exp(vol * np.sqrt(timeStep))
    # one step random walk (price decreases)
    d = np.exp(-vol * np.sqrt(timeStep))

    # risk neutral probability of an up move
    pu = (np.exp((intRate - divYield) * timeStep) - d) / (u - d)
    # risk neutral probability of a down move
    pd = 1 - pu

    # Tree is evaluated in two passes
    # In the first pass the probability weighted value of the stock price is calculated at each node
    # In the second pass the value of the option is calculated for each node given the stock price, backwards from the final time step
    # Note that the tree is recombinant, i.e. u+d = d+u

    # First we need an empty matrix big enough to hold all the calculated prices
    priceTree = np.full((totSteps, totSteps), np.nan) # matrix filled with NaN == missing values
    # Note that this tree is approx twice as big as needed because we only need one side of diagonal
    # We use the top diagonal for efficiency

    # Initialize with the current stock price, then loop through all the steps
    priceTree[0, 0] = currStockPrice

    for ii in range(1, totSteps):
        # vector calculation of all the up steps (show how the indexing works on line)
        priceTree[0:ii, ii] = priceTree[0:ii, (ii-1)] * u

        # The diagonal will hold the series of realizations that is always down
        # this is a scalar calculation
        priceTree[ii, ii] = priceTree[(ii-1), (ii-1)] * d

        #print("\n", priceTree)

    # Now we can calculate the value of the option at each node
    # We need a matrix to hold the option values that is the same size as the price tree
    # Note that size returns a vector of dimensions [r, c] which is why it can be passed as dimensions to nan
    optionValueTree = np.full_like(priceTree, np.nan)

    # First we calculate the terminal value
    if optionType == "CALL" and barrier > strikePrice:
        priceTree[priceTree[:,-1] >= barrier, -1] = 0
        optionValueTree[:, -1] = np.maximum(0, priceTree[:, -1] - strikePrice)
        # note the handy matrix shortcut syntax & max applied elementwise
    elif optionType == "PUT" and barrier < strikePrice:
        priceTree[priceTree[:,-1] <= barrier, -1] = 0
        optionValueTree[:, -1] = np.maximum(0, strikePrice - priceTree[:, -1])
        optionValueTree[optionValueTree[:, -1] == 100, -1] = 0
    else:
        # wherever possible, check that the input parameters are valid and raise an exception if not
        raise ValueError("Please check your input values")

    #print("\n", optionValueTree)

       

    oneStepDiscount = np.exp(-intRate * timeStep)    # discount rate for one step

    # Now we step backwards to calculate the probability weighted option value at every previous node
    # How many backwards steps?
    backSteps = priceTree.shape[1] - 1  # notice the shape function -> 1 is # of columns, which is last index + 1

    for ii in range(backSteps, 0, -1):
        optionValueTree[0:ii, ii-1] = \
            oneStepDiscount * \
            (pu * optionValueTree[0:ii, ii] \
            + pd * optionValueTree[1:(ii+1), ii])

        #if the option is american then you can convert at anytime, so the option value can never be less than the intrinsic value
        # if american:
        #     if optionType == "CALL":
        #         optionValueTree[0:ii, ii] = np.maximum(priceTree[0:ii, ii] - strikePrice, optionValueTree[0:ii, ii])
        #     else:
        #         optionValueTree[0:ii, ii] = np.maximum(strikePrice - priceTree[0:ii, ii], optionValueTree[0:ii, ii])

        #print("\n", optionValueTree)

    # After all that, the current price of the option will be in the first element of the optionValueTree
    optionPrice = optionValueTree[0, 0]

    # Delta is the difference between the up and down option prices at the first branch, divided by the price change
    # Note that time is not held constant in this way
    # the more steps the more closely the estimate will match "instantaneous"

    def D(x, y):
        d = (optionValueTree[x, y + 1] - optionValueTree[x + 1, y + 1]) / (priceTree[x, y + 1] - priceTree[x + 1, y + 1]);
        return d

    delta = D(0,0)

    gamma = (D(0,1) - D(1,1)) / (0.5 * (priceTree[0,2] - priceTree[2, 2]))

    # Note that theta is measured in the underlying units of timeStep, typically years
    theta = (optionValueTree[1, 2] - optionValueTree[0, 0]) / (2 * timeStep)
    theta = theta / 365

    return 'Barrier Option Price is {0:.4f}'.format(optionPrice)     # this return type is a "tuple" - a list of related data



if __name__ == "__main__":
    try:
        crr = CRROptionWGreeks_Barrier(100, 100, 120, 0.02, 0, 0.2, 500, 1, "CALL", False)
        print(crr)
    except ValueError as e:
        print("Type error: " + str(e))
    except Exception as e:
        print("Unknown error: " + str(e))

    # import BlackSholesPricer as bsp

    # bsch = bsp.EuropeanOption(100, 120, 0.02, 0, 0.2, 1, "CALL")
    # print(bsch)

    # for i in range(4):
    #    print((crr[i] - bsch[i]) / bsch[i])

    # vol = np.arange(0.1, 0.5, 0.005)

    # price = np.arange(80, 120, 0.25)

    # vval = np.empty_like(price)

    # i = 0
    # for v in vol:
    #     pr = CRROptionWGreeks(100, 100, 120, 0.02, 0, v, 100, 1, "PUT", False)
    #     vval[i] = pr[2]
    #     i += 1

    # i = 0
    # for p in price:
    #     pr = CRROptionWGreeks(p, 100, 80, 0.02, 0, 0.2, 500, 1, "PUT", False)
    #     vval[i] = pr[0]
    #     i += 1