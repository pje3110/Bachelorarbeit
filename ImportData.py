#%%
import numpy as np
import pandas
import importlib

import Calculations as calc
import OutputData as outp

importlib.reload(calc)
importlib.reload(outp)

##Import new dataset
##Assign variables for calculation
##Calculate elasticities
##Cluster products
##Output data
def importDataset(file):
    
    fFloat = open(file,"r")

    dataset = pandas.read_csv(fFloat, header=0, dtype={'ID_Article' : str, 'Sales' : 'f', 'Quantity' : 'f','Costs' : 'f', 'ID_Date' : 'str', 'Product_Group' : int, "Customer_Group" : 'str', "Branch_ID" : 'str'})
    #0: Artikel, 1: Preis, 2: Menge, 3: Kosten, 4: Quartal, 5: Produktgruppe, 6: Kundengruppe, 7: Filiale

    fFloat.close()

    ##Initialize variables
    numData = np.ones((len(dataset),3))                         #Only numerical values for calculation (price, quantity, cost)
    timeData = np.empty(len(dataset), dtype='datetime64[M]')    #Dates for all transactions
    articleData = np.zeros((len(numData),7))                    #Data for each article (price, quantity, cost, quarter, product group, firstPrice, firstQuantty)
    articleIDs = np.empty(len(numData), dtype='<U12')           #Article IDs for each line in articleData
    elasticity = np.zeros((len(articleData),2))                 #Calculated elasticities, first row for tagging value as used for optimization, second row for actual value
    branchIDs = np.empty(len(numData), dtype='<U32')            #Branch IDs for each transaction, get lost in aggregation

    ##Assign values from dataset to variables
    articleIDs[:] = dataset["ID_Article"]
    branchIDs[:] = dataset["Branch_ID"]
    timeData[:] = dataset["ID_Date"]
    numData[:,0] = dataset["Sales"] 
    numData[:,1] = dataset["Quantity"]
    numData[:,2] = dataset["Costs"]
    
    ##Average unit price from sales value
    numData[:,0] = np.round_(numData[:,0],2)/np.round_(numData[:,1],2)
    numData[:,2] = np.round_(numData[:,2],2)/np.round_(numData[:,1],2)
    numData[np.where(numData[:,0]<0.01),0] = 0.01
    numData[np.where(numData[:,2]<0.01),2] = 0.01

    ##Temporary array for aggregation to quarters, 
    ##Calculate quarters out of timeData,
    ##Split array into articles
    tempArray = np.zeros((len(numData), 5))
    tempArray[:,0:3] = numData
    tempArray[:,3] = ((abs(calc.getYear(timeData[:])) % 100) * 10) + calc.calculateQuarter(calc.getMonth(timeData[:]))
    tempArticles = np.split(tempArray, np.where(articleIDs[:-1] != articleIDs[1:])[0]+1)
    
    index = 0
    jumpVar = 0
    indexIncrease = 0
    targetQuarters = np.array([203])

    ##Iterate over each article for calculations
    for j in range(len(tempArticles)):

        ##Split articles into quarters
        tempQuarters = np.split(tempArticles[j], np.where(np.diff(tempArticles[j][:,3]))[0]+1)
        jumpVar = 0
        firstIndex = index

        ##Iterate over each quarter for calculations
        for l in range(len(tempQuarters)):
            jumpVar = len(tempQuarters[l]) 
            index += jumpVar
            articleData[index-1,0] = np.average(tempQuarters[l][:,0])                   #Price
            articleData[index-1,1] = np.sum(tempQuarters[l][:,1])                       #Quantity
            articleData[index-1,2] = np.average(tempQuarters[l][:,2])                   #Cost
            articleData[index-1,3] = (tempQuarters[l][0,3])                             #Quarter (204 -> Q4 2020)
            articleData[index-1,4] = dataset["Product_Group"][index-1 - indexIncrease]  #Product_Group
            articleIDs[index-1] = dataset["ID_Article"][index-1 - indexIncrease]        #Product ID

            ##Saves branch with most transactions for article/quarter
            uniqueIDs, position = np.unique(branchIDs[index-jumpVar:index], return_inverse=True)
            branchIDs[index-1] = uniqueIDs[np.bincount(position).argmax()]
            
            ##Temporary array for calculating elasticities,
            ##Prices of each transaction,
            ##Average quantity sold at price,
            ##Number of transactions at price,
            ##Save unique prices with according values
            elasticityArray = np.zeros((len(tempQuarters[l]),3))
            elasticityArray[:,0] = np.round(tempQuarters[l][:,0],2)
            
            for i in range(len(tempQuarters[l])):
                elasticityArray[i,1] = np.sum(tempQuarters[l][np.where(np.round(tempQuarters[l][:,0],2) >= elasticityArray[i,0]),1])
                elasticityArray[i,2] = np.shape(np.where(np.round(tempQuarters[l][:,0],2) == elasticityArray[i,0]))[1]        
            elasticityArray = np.unique(elasticityArray, axis=0)
            
            minPriceQuarter = np.min(elasticityArray[:,0])
            maxPriceQuarter = np.max(elasticityArray[:,0])
            articleData[index-1,5] = minPriceQuarter            #Min price in quarter
            articleData[index-1,6] = maxPriceQuarter            #Max price in quarter
            
            ##At least x transactions
            ##At least y price points
            elasticityArray = np.delete(elasticityArray, (elasticityArray[:,2] < 2), axis=0)
            if(len(elasticityArray) < 2):
                elasticity[index-1, 0:2] = 0,0
                continue

            ##Average price with weights based on number of transactions,
            ##Average quantity at price with weights based on distance to average price
            comparisonPrice = np.average(elasticityArray[:,0], weights = elasticityArray[:,2])
            comparisonQuantity = np.average(elasticityArray[:,1], weights = elasticityArray[:,2]) #weightsQuantity)
            elasticityArray[:,2] = 0 
            
            ##Prices and demand for elasticity calculation
            q0, q1 = np.log(comparisonQuantity), np.log(elasticityArray[:,1])       
            p0, p1 = comparisonPrice, elasticityArray[:,0]
            
            elasticityArray[:,2] = calc.calculateElasticityArray(p0,p1,q0,q1)
            elasticityArray = np.delete(elasticityArray, (elasticityArray[:,2] == 0), axis=0)
            
            if(len(elasticityArray) == 0):
                elasticity[index-1, 0:2] = 0,0
            else:    
                elasticity[index-1, 0:2] = 0, np.average(elasticityArray[:,2])

        elasticity[index-1,0] = 1
        averageArray = elasticity[firstIndex:index,1]
        averageArray = np.delete(averageArray, (averageArray == 0))

        ##Calculate percentage change of previous quarters to approximate values for target quarter
        ##Set elasticity
        for targetQuarter in targetQuarters:
            if(targetQuarter not in articleData[firstIndex:index, 3]):
                continue
                quarterNumber = calc.quarterDistance(181,targetQuarter)
                x = np.arange(1,quarterNumber,1)
                quarters = np.vstack([x, x + 181 - 1]).T
                quarters[:,1] += ((quarters[:,0]-1)/4).astype(int) * 6
                quarters = np.delete(quarters, (np.isin(quarters[:,1],  articleData[firstIndex:index, 3], invert=True)), axis = 0)
                quarters = np.delete(quarters, (np.logical_and(quarters[:,1]%10 != targetQuarter%10, (quarters[:,1]+1)%10 != targetQuarter%10)), axis = 0)
        
                n = 0
                while n < len(quarters)-1:
                    if(quarters[n,0]+1 != quarters[n+1,0]):
                        quarters = np.delete(quarters, n, axis=0)
                    else:
                        n += 2
                if((len(quarters) < 2) or (quarters[-1,1] != calc.lastQuarter(np.array([targetQuarter])))):
                    continue
        
                prices = np.delete(articleData[firstIndex:index,0], np.isin(articleData[firstIndex:index, 3], quarters[:,1], invert=True)) 
                quantities = np.delete(articleData[firstIndex:index,1], np.isin(articleData[firstIndex:index, 3], quarters[:,1], invert=True)) 
                costs = np.delete(articleData[firstIndex:index,2], np.isin(articleData[firstIndex:index, 3], quarters[:,1], invert=True)) 
                
                priceChange = np.zeros(len(prices)-1)
                quantityChange = np.zeros(len(quantities)-1)
                costChange = np.zeros(len(costs)-1)
                for i in range(0, len(prices)-1, 2):
                    priceChange[i] = (prices[i+1]-prices[i])/prices[i]
                    quantityChange[i] = (quantities[i+1]-quantities[i])/quantities[i]
                    costChange[i] = (costs[i+1]-costs[i])/costs[i]

                startingPrice = prices[-1] + prices[-1] * np.average(priceChange)
                startingQuantity = quantities[-1] + quantities[-1] * np.average(quantityChange)
                startingCost = costs[-1] + costs[-1] * np.average(costChange)


                articleData = np.insert(articleData, index, np.array((startingPrice, startingQuantity, startingCost, targetQuarter, articleData[index-1,4], 0, 0)), 0)
                if(len(averageArray) > 1):
                    elasticity = np.insert(elasticity, index, np.array((2, np.average(averageArray))), 0)
                else:
                    elasticity = np.insert(elasticity, index, np.array((0, 0)), 0)    
                articleIDs = np.insert(articleIDs, index, dataset["ID_Article"][index-1-indexIncrease])
                branchIDs = np.insert(branchIDs, index, "0")
                index += 1
                indexIncrease += 1
            else:
                if(len(averageArray > 1)):
                    location = np.where(np.logical_and(articleIDs[firstIndex:index] == articleIDs[index-1], articleData[firstIndex:index,3] == targetQuarter))[0] + firstIndex
                    elasticity[location,:] = 2, np.average(averageArray)
                                
    elasticity[:,1] = np.round(elasticity[:,1],1)
    
    ##print data count,
    ##Remove transactions except results of aggregation (marked with quarter)
    print("Transactions:\t", np.shape(articleIDs)[0], np.shape(articleData))
    articleIDs = np.delete(articleIDs, (articleData[:,3] == 0), axis=0)
    elasticity = np.delete(elasticity, (articleData[:,3] == 0), axis=0)
    branchIDs = np.delete(branchIDs, (articleData[:,3] == 0), axis=0)
    articleData = np.delete(articleData, (articleData[:,3] == 0), axis=0)
    print("Quarters:\t", np.shape(articleIDs)[0], np.shape(articleData))

    ##Output priceData of quarters with IDs, aggregated prices, costs, quantities, elasticity, branches
    file = "../priceData.csv"
    outp.outputPriceData(file, articleIDs, articleData, elasticity, branchIDs)

    ##Remove all data except last quarter for optimization
    articleData = np.delete(articleData, (elasticity[:,0] < 1), axis=0)
    articleIDs = np.delete(articleIDs, (elasticity[:,0] < 1), axis=0)
    elasticity = np.delete(elasticity, (elasticity[:,0] < 1), axis=0)

    ##Clustering articles based on quantiles
    articleCluster = np.zeros(len(articleIDs))
    clusterArticleData = np.zeros((len(articleData),3))
    clusterElasticity = np.zeros((len(elasticity),2))
    clusterElasticity[:,0] = np.round(elasticity[:,1],2)

    clusterProductGroups = calc.getFirstDigits(articleData[:,4], 1)
    articleCluster = clusterProductGroups * 1000
    
    clusterElasticity[np.where(clusterElasticity[:,0]>=np.quantile(clusterElasticity[np.where(clusterElasticity[:,0] > 0),0],0.8)),1] = 5
    clusterElasticity[np.where(clusterElasticity[:,0]<np.quantile(clusterElasticity[np.where(clusterElasticity[:,0] > 0),0],0.8)),1] = 4
    clusterElasticity[np.where(clusterElasticity[:,0]<np.quantile(clusterElasticity[np.where(clusterElasticity[:,0] > 0),0],0.6)),1] = 3
    clusterElasticity[np.where(clusterElasticity[:,0]<np.quantile(clusterElasticity[np.where(clusterElasticity[:,0] > 0),0],0.4)),1] = 2
    clusterElasticity[np.where(clusterElasticity[:,0]<np.quantile(clusterElasticity[np.where(clusterElasticity[:,0] > 0),0],0.2)),1] = 1
    
    occurences = np.zeros((len(clusterProductGroups)), dtype=object)
    quantileValues = np.zeros((len(clusterProductGroups),8))

    for i in range(len(clusterProductGroups)):
        occurences[i] = np.flatnonzero(calc.getFirstDigits(articleData[i,4],1) == clusterProductGroups)
    
    for i in range(len(quantileValues)):
        quantileValues[i,0:4] = np.quantile(articleData[occurences[i],0],0.8), np.quantile(articleData[occurences[i],0],0.6), np.quantile(articleData[occurences[i],0],0.4), np.quantile(articleData[occurences[i],0],0.2) 
        quantileValues[i,4:8] = np.quantile(articleData[occurences[i],1],0.8), np.quantile(articleData[occurences[i],1],0.6), np.quantile(articleData[occurences[i],1],0.4), np.quantile(articleData[occurences[i],1],0.2) 
        
    clusterArticleData[np.where(articleData[:,0:2]>=quantileValues[:,[0,4]])] = 5
    clusterArticleData[np.where(articleData[:,0:2]<quantileValues[:,[0,4]])] = 4
    clusterArticleData[np.where(articleData[:,0:2]<quantileValues[:,[1,5]])] = 3
    clusterArticleData[np.where(articleData[:,0:2]<quantileValues[:,[2,6]])] = 2
    clusterArticleData[np.where(articleData[:,0:2]<quantileValues[:,[3,7]])] = 1
    articleCluster = articleCluster + clusterElasticity[:,1] * 100 + clusterArticleData[:,0] * 10 + clusterArticleData[:,1] * 1
    
    ##Output articleData with all IDs, product groups and clusters
    file = "../articleData.csv"
    outp.outputArticleData(file, articleIDs, articleData, articleCluster)

    ##Remove articles not in target quarter or with no elasticity 
    elasticity[np.where(elasticity[:,0] < 2),1] = 0
    articleIDs = np.delete(articleIDs, (elasticity[:,1] == 0), axis=0)
    articleCluster = np.delete(articleCluster, (elasticity[:,1] == 0), axis=0)
    articleData = np.delete(articleData, (elasticity[:,1] == 0), axis=0)
    elasticity = np.delete(elasticity, (elasticity[:,1] == 0), axis=0)
    print("Articles:\t", np.shape(articleIDs)[0])
    
    print("Clusters:\t", np.shape(np.unique(articleCluster))[0])
    
    ##Initialize variables for algorithm
    calcData = np.copy(articleData)                                                 ##Temporary storage for calculated results
    optimizedData = np.zeros((len(articleData),3))                                  ##Permanent storage for optimization results
    productGroups = np.split(articleData, np.where(np.diff(articleData[:,4]))[0]+1) ##Articles split into product groups
    qTables = []                                                                    

    ##Return variables
    print("Import finished\n")
    return articleData, articleIDs, articleCluster, calcData, elasticity, optimizedData, productGroups, qTables


##Import existing dataset
##Assign variables for calculation
def importPreparedDataset():
    
    file = "../priceData.csv"
    file2 = "../articleData.csv"
    
    fFloat = open(file,"r")

    dataset = pandas.read_csv(fFloat, header=0, dtype={'Article_ID' : str, 'Price' : 'f', 'Quantity' : 'f','Cost' : 'f', 'Quarter' : 'str', 'Elasticity' : 'f'})
    
    fFloat.close()

    fFloat2 = open(file2,"r")

    dataset2 = pandas.read_csv(fFloat2, header=0, dtype={'Article_ID' : str, 'Product_Group' : str, 'Cluster' : int})
    
    fFloat2.close()

    articleData = np.zeros((len(dataset),5))
    elasticity = np.zeros((len(articleData),2))
    articleIDs = np.empty(len(articleData), dtype='<U12')
    articleIDs = dataset["Article_ID"].to_numpy(dtype='<U12')
    articleData[:,0] = dataset["Price"]
    articleData[:,1] = dataset["Quantity"]
    articleData[:,2] = dataset["Cost"]
    articleData[:,3] = dataset["Quarter"]
    elasticity[:,1] = dataset["Elasticity"]
    
    elasticity[np.where(articleIDs[:-1] != articleIDs[1:])[0],0] = 1
    if(elasticity[-1,1] != 0):
        elasticity[-1,0] = 1
    
    targetQuarter = 203
    elasticity[np.where(articleData[:,3] != targetQuarter),1] = 0

    print("Quarters:\t", np.shape(articleIDs)[0], np.shape(articleData))
    articleData = np.delete(articleData, (elasticity[:,0] == 0), axis=0)
    articleIDs = np.delete(articleIDs, (elasticity[:,0] == 0), axis=0)
    elasticity = np.delete(elasticity, (elasticity[:,0] == 0), axis=0)
    
    articleData[:,4] = dataset2["Product_Group"]
    articleCluster = dataset2["Cluster"].to_numpy(dtype='int')

    articleIDs = np.delete(articleIDs, (elasticity[:,1] == 0), axis=0)
    articleData = np.delete(articleData, (elasticity[:,1] == 0), axis=0)
    elasticity = np.delete(elasticity, (elasticity[:,1] == 0), axis=0)
    print("Articles:\t", np.shape(articleIDs)[0])
    
    calcData = np.copy(articleData)
    optimizedData = np.zeros((len(articleData),3))
    productGroups = np.split(articleData, np.where(np.diff(articleData[:,4]))[0]+1)
    qTables = []
    
    print("Import finished")
    return articleData, articleIDs, articleCluster, calcData, elasticity, optimizedData, productGroups, qTables


#%%