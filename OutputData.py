import numpy as np
import importlib

import Calculations as calc

importlib.reload(calc)

##Output results of optimization
def outputResults(file, articleData, articleIDs, elasticity, optimizedData):

    tempArticleIDs = np.zeros((len(articleIDs),2), dtype='<U12')
    tempArticleIDs[:,0] = articleIDs[:]

    header = "Article_ID,Opt_Price,Opt_Quantity,Opt_Cost,Quarter,Elasticity"
    finalData = np.concatenate((tempArticleIDs[:,0:1], optimizedData[:,0:3], articleData[:,3:4].astype(int), elasticity[:,1:2]), axis=1)
    
    np.savetxt(file, finalData, fmt='%s', delimiter=',', newline='\n', header=header, comments='')

##Output quarterly transactional article data
def outputPriceData(file, allArticleIDs, allArticleData, allElasticity, branchIDs):
    
    tempArticleIDs = np.zeros((len(allArticleIDs),2), dtype='<U12')
    tempArticleIDs[:,0] = allArticleIDs[:]
    
    tempBranchIDs = np.zeros((len(branchIDs),2), dtype='<U32')
    tempBranchIDs[:,0] = branchIDs[:]
    
    tempArticleQuarters = allArticleData[:,3:4].astype(int)

    header = "Article_ID,Price,Quantity,Cost,Quarter,Elasticity,Branch_ID,MinPrice,MaxPrice"
    finalData = np.concatenate((tempArticleIDs[:,0:1], allArticleData[:,0:3], tempArticleQuarters[:,0:1], allElasticity[:,1:2], tempBranchIDs[:,0:1], allArticleData[:,5:7]), axis=1)

    np.savetxt(file, finalData, fmt='%s', delimiter=',', newline='\n', header=header, comments='')

##Output dimensional article information
def outputArticleData(file, allArticleIDs, allArticleData, articleClusters):
    allArticleData = allArticleData[:].astype(int)
    tempArticleIDs = np.zeros((len(np.unique(allArticleIDs)),2), dtype='<U12')
    tempArticleIDs[:,0], indices = np.unique(allArticleIDs, return_index=True)
    tempArticleClusters = np.zeros((len(tempArticleIDs),2), dtype='>i4')
    tempArticleClusters[:,0] = articleClusters[indices].astype(int)

    header = "Article_ID,Product_Group,Cluster"
    finalData = np.concatenate((tempArticleIDs[:,0:1], allArticleData[indices,4:5], tempArticleClusters[:,0:1]), axis=1)

    np.savetxt(file, finalData, fmt='%s', delimiter=',', newline='\n', header=header, comments='')