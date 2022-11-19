#%%
import numpy as np

#---Time Calculations---

##Calculate Month from Date
def getMonth(datetime):
    return datetime.astype(int) % 12 + 1

##Calculate Year from Date
def getYear(datetime):
    return datetime.astype('datetime64[Y]').astype(int) + 1970

##Calculate Quarter from Date
def calculateQuarter(month):
    return np.ceil(month/3)

##Calculate next Quarter
def nextQuarter(quarter):
    quarter = quarter + 1
    quarter[np.where((quarter%10)>4)] = quarter[np.where((quarter%10)>4)] + 6 
    return quarter

##Calculate next Quarter
def lastQuarter(quarter):
    quarter = quarter - 1
    quarter[np.where((quarter%10)<1)] = quarter[np.where((quarter%10)<1)] - 6 
    return quarter

##Calculate Distance between Quarters
def quarterDistance(start, target):
    return int(((np.round(target/10) - np.round(start/10)) * 4) + (target%10 - start%10) + 1)


#---Economical Calculations---

##Calculate total optimized Profit of Dataset
def calculateOptimizedProfit(optimizedData):
    profit = np.sum((optimizedData[:,0] - optimizedData[:,2]) * optimizedData[:,1])
    return profit

##Calculate total Profit of Dataset
def calculateTotalProfit(calcData):
    profit = np.sum((calcData[:,0] - calcData[:,2]) * calcData[:,1])
    return profit

##Calculate Profit of single Article
def calculateProfit(index, calcData):
    profit = ((calcData[index,0] - calcData[index,2]) * calcData[index,1])
    return profit

##Calculate resulting Quantity of Price Change
def calculateQuantity(index, newPrice, calcData, elasticity):
    articleElasticity = elasticity[index,1]    
    oldPrice = calcData[index, 0]
    oldQuantity = calcData[index, 1]
    priceChange = calculatePercentChange(oldPrice,newPrice)
    newQuantity = oldQuantity + (-articleElasticity * priceChange * oldQuantity)
    if(newQuantity < 0):
        newQuantity = 0
    elif(newQuantity > (oldQuantity * 2)):
        newQuantity = oldQuantity * 2
    return newQuantity

##Calculate and set resulting Quantity of Price Change
def calculateSetQuantity(index, newPrice, calcData, elasticity):
    calcData[index,0:2] = newPrice, calculateQuantity(index, newPrice, calcData, elasticity)
    return calcData

##Calculate percentage Change
def calculatePercentChange(old, new):
    if(old != 0):
        percentChange = (new - old)/old
        return percentChange
    else:
        return 0


#---Elasticity Calculations---

##Calculate Elasticity of Values in Array
def calculateElasticityArray(p0, p1, q0, q1):
    result = np.zeros(len(p1))
    p = ((p1-p0)/p0)
    q = ((q1-q0)/q0)
    divZeroPrevent = np.where(p != 0)
    result[divZeroPrevent] = q[divZeroPrevent]/p[divZeroPrevent]
    result[np.where((result > -0.1) | (result < -5) | (p == 0))] = 0
    return abs(result)


#---Utility Calculations---

##Calculate first Digits of a Value
def getFirstDigits(value, n):
    result = (value // 10 ** ((np.log10(value)).astype(int) - n + 1))
    return result


#---Asserts---

def runAsserts():
    testTimes = np.array(["2020-01","2021-02","2022-12"], dtype='datetime64[M]')

    results = np.array([1,2,12])
    assert np.array_equal(getMonth(testTimes), results)

    results = np.array([2020,2021,2022])
    assert np.array_equal(getYear(testTimes), results)

    testQuarters = np.array([191,192,193,194])
    assert calculateQuarter(1) == 1
    assert calculateQuarter(3) == 1
    assert calculateQuarter(9) == 3

    results = np.array([192,193,194,201])
    assert np.array_equal(nextQuarter(testQuarters), results)

    results = np.array([184,191,192,193])
    assert np.array_equal(lastQuarter(testQuarters), results)

    assert quarterDistance(191,204) == 8

    testData = np.array(([3,2,1],[2,3,1]))
    assert calculateOptimizedProfit(testData) == 7
    assert calculateTotalProfit(testData) == 7
    assert calculateProfit(0, testData) == 4
    assert calculateQuantity(0,1,testData,np.array(([1,2],[1,3]))) == 4

    assert calculatePercentChange(10,5) == -0.5
    assert calculatePercentChange(5,10) == 1

    p0 = 6
    p1 = np.array([5,2,4,8,10,7])
    q0 = 20
    q1 = np.array([100,35,30,15,10,30])

    results = np.array([0,1.125,1.5,0.75,0.75,0])
    assert np.array_equal(calculateElasticityArray(p0,p1,q0,q1),results) 

    assert getFirstDigits(50,1) == 5
    assert getFirstDigits(50,2) == 50
