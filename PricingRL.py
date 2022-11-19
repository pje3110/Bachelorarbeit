#%% 
import math
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
articleData, articleIDs, articleCluster, calcData, elasticity, optimizedData, productGroups, qTables = imp.importDataset(file)

##Import existing dataset
#articleData, articleIDs, articleCluster, calcData, elasticity, optimizedData, productGroups, qTables = imp.importPreparedDataset()

def reset():
    global calcData
    calcData = np.copy(articleData)

def resetRow(index):
    global calcData
    calcData[index,:] = np.copy(articleData[index,:])


#-----------------------
#-----Reinforcement-----
#-----------------------

class ReinforcementLearning:

    ##Initialize object
    def __init__(self, article = -1):
        if(article != -1):
            self.article = article
        else:
            self.article = 0

    def setArticle(self, article):
        self.article = article

    ##Set article, initialize variables and create table
    def prepareAlgorithm(self, article):
        self.setArticle(article)
        self.initializeVariables()
        self.resetTable()

    ##Initialize variables for training, limits
    def initializeVariables(self, basePrice = 0, upperLimit = 0, lowerLimit = 0):
        
        self.numEpisodes = 50
        self.numSteps = 300

        priceLimit = 0.2
        stepSize = 0.005
        
        if(basePrice > 0):
            self.originalPrice = basePrice
        else:
            self.originalPrice = articleData[self.article,0]
        if(upperLimit > 0):
            self.upperLimit = upperLimit + upperLimit * priceLimit
        else:
            self.upperLimit = self.originalPrice + self.originalPrice * priceLimit
        if(lowerLimit > 0):
            self.lowerLimit = lowerLimit - lowerLimit * priceLimit
        else:
            self.lowerLimit = self.originalPrice - self.originalPrice * priceLimit

        self.stepPrice = self.originalPrice * stepSize
        self.numStates = math.floor(((self.upperLimit) - (self.lowerLimit)) / (self.stepPrice)) + 1
        self.numActions = 3
        self.priceStates = np.arange(self.lowerLimit, self.upperLimit+self.stepPrice, self.stepPrice)
        
    ##Create or reset table for storing training results
    def resetTable(self):
        self.qTable = np.zeros((self.numStates, self.numActions))
        
    ##State transition with action (raise, lower, keep)
    def execute(self, currentState, action):
        global calcData, articleData
        nextState = currentState
        reward = 0  
        finishBool = False
        self.lastPrice = self.currentPrice
        
        match action:
            case 0:
                if(currentState < (self.numStates - 1)):
                    nextState += 1
                    self.currentPrice += self.stepPrice
                else:
                    nextState = currentState
                    reward = -100
            case 1:
                if(currentState > 0):
                    nextState -= 1
                    self.currentPrice -= self.stepPrice
                else:
                    nextState = currentState
                    reward = -100
            case 2:
                nextState = currentState
            case _:
                print("Error, invalid action")
                raise SystemExit
        
        ##Early break if no state changes
        if(nextState == currentState):
            self.repetitionCounter += 1
            if(self.repetitionCounter > self.breakValue):
                finishBool = True
        else:
            self.repetitionCounter = 0

        ##Calculate reward based on profit
        calcData = calc.calculateSetQuantity(self.article, self.currentPrice, calcData, elasticity)
        newProfit = calc.calculateProfit(self.article, calcData)

        if((newProfit > self.maxProfit) and (reward == 0)):
            self.maxProfit = newProfit
            self.optimalPrice = self.currentPrice

        if(self.currentProfit == 0):
            self.currentProfit = 0.01
        reward = (newProfit - self.currentProfit)/self.currentProfit

        self.currentProfit = newProfit
        resetRow(self.article)
        return nextState, reward, finishBool


    #Train Aagorithm with article
    def fit(self,article = -1, qTable = np.array([1])):
        if(article > -1):
            self.article = article
        if(len(qTable) > 1):
            self.qTable = qTable

        minEpsilon, maxEpsilon = 0.01,1
        epsilonDecay = 0.01
        epsilon = 1         #Exploration rate
        alpha = 0.1         #Learning rate
        gamma = 0.8         #Discount rate             
        
        self.breakValue = 5
        priceTotal = 0
        self.originalPrice = articleData[self.article,0]
        for episode in range(self.numEpisodes):
            
            reset()
            self.currentPrice = self.originalPrice
            self.optimalPrice = 0
            self.currentProfit = calc.calculateProfit(self.article, calcData)
            self.maxProfit = 1
            self.lastProfit = 1
            currentState = (np.abs(self.priceStates - self.originalPrice)).argmin() 
            self.repetitionCounter = 0
            finishBool = False
            rewardCount = 0
            
            for step in range(self.numSteps):
                
                resetRow(self.article)
                epsilonDecider = np.random.random()
                if(epsilonDecider > epsilon):
                    action = np.argmax(self.qTable[currentState,:])
                else:
                    action = np.random.choice(self.numActions)
                nextState, reward, finishBool = self.execute(currentState,action)
                self.qTable[currentState, action] = self.qTable[currentState, action] * (1 - alpha) + alpha * (reward + gamma * np.max(self.qTable[nextState,:]))
                currentState = nextState
                rewardCount += reward
                if(finishBool):
                    break

            epsilon = minEpsilon + (maxEpsilon - minEpsilon) * np.exp(-epsilonDecay*episode)
            if(episode > (self.numEpisodes * 0.8)):
                priceTotal += self.optimalPrice
        reset()
        return self.qTable
    
    ##Exploit knowledge for article with training table
    def optimize(self, article, qTable):
        global calcData
        self.article = article
        self.qTable = qTable
        self.numEpisodes = 1
        self.numSteps = 50    
        self.breakValue = 1
        priceTotal = 0
        self.originalProfit = calc.calculateProfit(self.article, calcData)
        self.originalPrice = articleData[self.article,0]

        for episode in range(self.numEpisodes):
            
            reset()
            self.currentPrice = self.originalPrice
            self.optimalPrice = 0
            self.currentProfit = 1
            self.maxProfit = 1
            currentState = (np.abs(self.priceStates - self.originalPrice)).argmin() 
            self.qTable[currentState,2] = 0
            self.repetitionCounter = 0
            self.lastAction = -1
            finishBool = False
            
            for step in range(self.numSteps):
                resetRow(self.article)
                action = np.argmax(self.qTable[currentState,:])
                nextState, _, finishBool = self.execute(currentState,action)
                currentState = nextState

                if(finishBool):
                    break
                
            priceTotal += self.optimalPrice    

        priceTotal /= self.numEpisodes
        calcData = calc.calculateSetQuantity(self.article, priceTotal, calcData, elasticity)
        newProfit = calc.calculateProfit(article, calcData)
        reset()
        if(newProfit > self.originalProfit):
            optimizedData[self.article,0] = priceTotal
            optimizedData[self.article,1] = calc.calculateQuantity(self.article,priceTotal, calcData, elasticity)
            optimizedData[self.article,2] = articleData[self.article,2]
        else:
            optimizedData[self.article,0] = self.originalPrice
            optimizedData[self.article,1] = calc.calculateQuantity(self.article,self.originalPrice, calcData, elasticity)
            optimizedData[self.article,2] = articleData[self.article,2]


#---------
#---Run---

def runAlgorithm():
    rl = ReinforcementLearning()
    for i in range(len(articleData)):
        rl.prepareAlgorithm(i)
        qTable1 = rl.fit()
        rl.optimize(article = i, qTable = qTable1)
    print("Done optimizing!\n")

##Optimize by cluster
def optimizeCluster():
    global articleCluster, articleIDs, articleData
    sortOrder = articleCluster.argsort()
    sortedCluster = np.sort(articleCluster)
    articleIDs = articleIDs[sortOrder]
    articleData = articleData[sortOrder,:]
    clusterGroups = np.split(articleData, np.where(np.diff(sortedCluster))[0]+1)
    rl = ReinforcementLearning()
    article = 0
    for h in range(len(clusterGroups)):
        upperLimit = np.max(clusterGroups[h][:,0])
        lowerLimit = np.min(clusterGroups[h][:,0])
        basePrice = np.average(clusterGroups[h][:,0])

        rl.setArticle(article)
        rl.initializeVariables(basePrice, upperLimit, lowerLimit)
        rl.resetTable()

        qTable = rl.fit(article)
        for j in range(len(clusterGroups[h])):
            rl.optimize(article, qTable)  
            article += 1
    qTables.append(rl.qTable)
    
runAlgorithm()
#optimizeCluster()


file = "../optimizedDataRL.csv"
outp.outputResults(file, articleData, articleIDs, elasticity, optimizedData)

#%%