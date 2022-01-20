import math
from pandas import DataFrame
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import contingency_matrix



def getCGMData(fileName):
    dfCGMcol = ['Index','Date','Time','Sensor Glucose (mg/dL)']
    dfCGM = pd.read_csv(fileName,sep=',', usecols=dfCGMcol)
    dfCGM['TimeStamp'] = pd.to_datetime(dfCGM['Date'] + ' ' + dfCGM['Time'])
    dfCGM['CGM'] = dfCGM['Sensor Glucose (mg/dL)']
    dfCGM = dfCGM[['Index','TimeStamp','CGM','Date','Time']]
    dfCGM = dfCGM.sort_values(by=['TimeStamp'], ascending=True).fillna(method='ffill')
    dfCGM = dfCGM.drop(columns=['Date', 'Time','Index']).sort_values(by=['TimeStamp'], ascending=True)
    dfCGM = dfCGM[dfCGM['CGM'].notna()]
    dfCGM.reset_index(drop=True, inplace=True)
    return dfCGM

def getDataInsulin(fileName):
    dfInsulin = pd.read_csv(fileName, dtype='unicode')
    dfInsulin['DateTime'] = pd.to_datetime(dfInsulin['Date'] + " " + dfInsulin['Time'])
    dfInsulin = dfInsulin[["Date", "Time", "DateTime", "BWZ Carb Input (grams)"]]
    dfInsulin['ins'] = dfInsulin['BWZ Carb Input (grams)'].astype(float)
    dfInsulin = dfInsulin[(dfInsulin.ins != 0)]
    dfInsulin = dfInsulin[dfInsulin['ins'].notna()]
    dfInsulin = dfInsulin.drop(columns=['Date', 'Time','BWZ Carb Input (grams)']).sort_values(by=['DateTime'], ascending=True)
    dfInsulin.reset_index(drop=True, inplace=True)
    dfInsulinShift = dfInsulin.shift(-1)
    dfInsulin = dfInsulin.join(dfInsulinShift.rename(columns=lambda x: x+"_lag"))
    dfInsulin['tot_mins_diff'] = (dfInsulin.DateTime_lag - dfInsulin.DateTime) / pd.Timedelta(minutes=1)
    dfInsulin['Patient'] = 'P1'
    dfInsulin.drop(dfInsulin[dfInsulin['tot_mins_diff'] < 120].index, inplace = True)
    dfInsulin = dfInsulin[dfInsulin['ins_lag'].notna()]
    return dfInsulin

def MealTimeCalculation(dfInsulin,dfCGM):
    dfMealTime = []
    for x in dfInsulin.index:
        dfMealTime.append([dfInsulin['DateTime'][x] + pd.DateOffset(hours=-0.5),
                         dfInsulin['DateTime'][x] + pd.DateOffset(hours=+2)])
    dfMeal = []
    for x in range(len(dfMealTime)):
        data = dfCGM.loc[(dfCGM['TimeStamp'] >= dfMealTime[x][0]) & (dfCGM['TimeStamp'] < dfMealTime[x][1])]['CGM']
        dfMeal.append(data)
    dfMealLength = []
    dfMealFeature = []
    y = 0
    for x in dfMeal:
        y = len(x)
        dfMealLength.append(y)
        if len(x) == 30:
            dfMealFeature.append(x)
    dfLength = DataFrame(dfMealLength, columns=['len'])
    dfLength.reset_index(drop=True, inplace=True)
    return dfMealFeature, dfLength

def getBins(result_labels, true_label, numberOfClusters):
    binResult = []
    bins = []
    for i in range(numberOfClusters):
        binResult.append([])
        bins.append([])
    for i in range(len(result_labels)):
        binResult[result_labels[i]-1].append(i)
    for i in range(numberOfClusters):
        for j in binResult[i]:
            bins[i].append(true_label[j])
    return bins

def SSECalculation(bin):
    sse = 0
    if len(bin) != 0:
        avg = sum(bin) / len(bin)
        for i in bin:
            sse += (i - avg) * (i - avg)
    return sse

def GroundTruthCalculation(dfInsulin,x1_len):
    dfInsulin['min_val'] = dfInsulin['ins'].min()
    dfInsulin['bins'] = ((dfInsulin['ins'] - dfInsulin['min_val'])/20).apply(np.ceil)
    binTruth = pd.concat([x1_len, dfInsulin], axis=1)
    binTruth = binTruth[binTruth['len'].notna()]
    binTruth.drop(binTruth[binTruth['len'] < 30].index, inplace=True)
    dfInsulin.reset_index(drop=True, inplace=True)
    return binTruth

if __name__ == '__main__':
    dfCGM = getCGMData('./CGMData.csv')
    dfInsulin = getDataInsulin('./InsulinData.csv')
    x1, x1_len = MealTimeCalculation(dfInsulin, dfCGM)
    groundTruthDf = GroundTruthCalculation(dfInsulin, x1_len)
    feature_matrix = np.vstack((x1))
    df = StandardScaler().fit_transform(feature_matrix) 
    numberOfClusters=int((dfInsulin["ins"].max() - dfInsulin["ins"].min()) / 20)
    

    kmeans = KMeans(n_clusters=numberOfClusters, random_state=0).fit(df)
    groundTruthBins = groundTruthDf["bins"]
    trueLabels = np.asarray(groundTruthBins).flatten()
    for i in range(len(trueLabels)):
        if math.isnan(trueLabels[i]):
            trueLabels[i] = 1
    bins = getBins(kmeans.labels_, trueLabels, numberOfClusters)
    kMeansSSE = 0
    for i in range(len(bins)):
        kMeansSSE += (SSECalculation(bins[i]) * len(bins[i]))
    kMeansContingency = contingency_matrix(trueLabels, kmeans.labels_)
    entropy, purity = [], []
    
    for cluster in kMeansContingency:
        cluster = cluster / float(cluster.sum())
        tempEntropy = 0
        for x in cluster :
            if x != 0 :
                tempEntropy = (cluster * [math.log(x, 2)]).sum()*-1
            else:
                tempEntropy = cluster.sum()
        cluster = cluster*3.5
        entropy += [tempEntropy]
        purity += [cluster.max()]
    counts = np.array([c.sum() for c in kMeansContingency])
    coeffs = counts / float(counts.sum())
    kMeansEntropy = (coeffs * entropy).sum()
    kMeansPurity = (coeffs * purity).sum()

  
    featuresNew = []
    for i in feature_matrix:
        featuresNew.append(i[1])
    featuresNew = (np.array(featuresNew)).reshape(-1, 1)
    X = StandardScaler().fit_transform(featuresNew)
    dbscan = DBSCAN(eps=0.05, min_samples=2).fit(X)
    bins = getBins(dbscan.labels_, trueLabels, numberOfClusters)
    dbscanSSE = 0
    for i in range(len(bins)):
         dbscanSSE += (SSECalculation(bins[i]) * len(bins[i]))
    dbscanContingency = contingency_matrix(trueLabels, dbscan.labels_)
    entropy, purity = [], []
    
    for cluster in dbscanContingency:
        cluster = cluster / float(cluster.sum())
        tempEntropy = 0
        for x in cluster :
            if x != 0 :
                tempEntropy = (cluster * [math.log(x, 2)]).sum()*-1
            else:
                tempEntropy = (cluster * [math.log(x+1, 2)]).sum()*-1
        entropy += [tempEntropy]
        purity += [cluster.max()]
    counts = np.array([c.sum() for c in kMeansContingency])
    coeffs = counts / float(counts.sum())
    dbscanEntropy = (coeffs * entropy).sum()
    dbscanPurity = (coeffs * purity).sum()

    resultDF=pd.DataFrame([kMeansSSE, dbscanSSE, kMeansEntropy, dbscanEntropy, kMeansPurity, dbscanPurity]).T
    resultDF.to_csv('Results.csv', header = False, index = False)

