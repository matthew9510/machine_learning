import numpy as np
import matplotlib.pyplot as plt

# Starting codes for the HA2 of CS596

# Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# Ground-truth Cashier
groundUnitPrice = np.array([20, 25, 8])  # for fish, chip, and ketchup, respectively

# step 1: initialize your guess on the unit prices of fish, chip and ketchup.
estimatedUnitPrice = np.array([1, 1, 1])  # initial unit prices.
# MAX_POSSIBLE_UNIT_PRICE = 50
# estimatedUnitPrice = np.random.randint(MAX_POSSIBLE_UNIT_PRICE, size=3) # choose random initial guesses

# PLACEHOLDER_1#start: set your own stopping conditions and learning rate
# condition 1: maximal iterations, stop.
MAX_ITERATION = 500
# condition 2: if the difference between your prediction and the cashier's price is smaller than a threshold, stop.
MIN_DELTA = 0.01
# learning rate
ALPHA = .01  # 1e-3
# PLACEHOLDER_1#end

# Y coordinates for plotting
deltaHistory = []

# step 2: iterative method
for i in range(0, MAX_ITERATION):
    # order a meal (simulating training data)
    randomMealPortions = np.random.randint(10, size=3) + 1

    # calculate the estimated price
    expectedTotalPrice = np.sum(estimatedUnitPrice * randomMealPortions)

    # calculate cashier/true price;
    cashierPrice = np.sum(groundUnitPrice * randomMealPortions)

    # %%%PLACEHOLDER_2#start

    # calculate current error
    delta = expectedTotalPrice - cashierPrice

    # append iterError to the history array
    deltaHistory.append(abs(delta))

    # update unit prices
    
    estimatedUnitPrice = estimatedUnitPrice - (delta * ALPHA * randomMealPortions)

    # %%%%PLACEHOLDER_2#end

    # check stop conditions
    if abs(delta) < MIN_DELTA:
        break

    print('iteration:{}, delta:{}'.format(i, abs(delta)))

# step 3: evaluation
error = np.mean(abs(estimatedUnitPrice - groundUnitPrice))
print('estimation error:{}'.format(error))

# visualize convergence curve: error v.s. iterations

plt.plot(range(0, len(deltaHistory)), deltaHistory)
plt.xlabel('iteration (cnt:{})'.format(len(deltaHistory)))
plt.ylabel('abs(delta)')
plt.title('Final:{}  est err:{}  actl Δ:{}'.format(['%.4f' % elem for elem in estimatedUnitPrice], round(error, 4),
                                                   round(delta, 4)))
plt.show()
