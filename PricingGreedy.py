#%% 
#Import started
import numpy as np
import importlib

import Calculations as calc
import ImportData as imp
import OutputData as outp

importlib.reload(calc)
importlib.reload(imp)
importlib.reload(outp)

calc.runAsserts()

np.random.seed(42)

file = "../dataset.csv"

##Import new dataset    
articleData, articleIDs, _, calcData, elasticity, optimizedData, _, _ = imp.importDataset(file)

##Import existing dataset
#articleData, articleIDs, _, calcData, elasticity, optimizedData, _, _ = imp.importPreparedDataset()

def reset():
    global calcData
    calcData = np.copy(articleData)

def resetRow(index):
    global calcData
    calcData[index,:] = np.copy(articleData[index])

#------------------------
#--Optimization method---

##Greedy price optimization
def greedyOptimal(index):
    global calcData
    reset()
    maxProfit, lastProfit, maxPrice, maxQuantity, errorCount, i = 0,0,0,0,0,1
    currentPrice = calcData[index,0]
    currentProfit = calc.calculateProfit(index, calcData)
    originalProfit = currentProfit
    originalPrice = currentPrice
    originalQuantity = calcData[index, 1]
    addition = 0.005*currentPrice
    upperPriceLimit = 0.2
    lowerPriceLimit = 0.2
    maxSteps = (lowerPriceLimit/(addition/currentPrice))+1
    changePoint = (upperPriceLimit/(addition/currentPrice))

    while (errorCount < 3) and (i <= maxSteps):
        calcData = calc.calculateSetQuantity(index, currentPrice, calcData, elasticity)
        currentProfit = calc.calculateProfit(index, calcData)
        
        if(lastProfit > currentProfit):
            errorCount += 1
        
        if(currentProfit > maxProfit):
            maxProfit = currentProfit
            maxPrice, maxQuantity = calcData[index,0:2]

        if(errorCount > 1 or (i > changePoint and addition > 0)):
            if(addition > 0):
                currentPrice = originalPrice
                currentProfit = originalProfit
                addition *= -1
                errorCount = 0                          
                i = 1
            else:
                resetRow(index)
                break
        
        currentPrice = originalPrice + i * addition

        lastPrice = currentPrice
        lastProfit = currentProfit
        resetRow(index)
        i += 1
    optimizedData[index,0] = maxPrice
    optimizedData[index,1] = maxQuantity
    optimizedData[index,2] = articleData[index,2]


#-------
#--Run--
 
def runAlgorithm():
    for i in range(len(articleData)):
        greedyOptimal(i)
    print("Done optimizing!\n")

runAlgorithm()


file = "../optimizedDataRL.csv"
outp.outputResults(file, articleData, articleIDs, elasticity, optimizedData)

#%%