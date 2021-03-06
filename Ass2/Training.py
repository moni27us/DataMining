# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:32:35 2020

@author: depri
"""
import pandas as pd
import datetime
from datetime import timedelta
import math
import numpy as np
from sklearn.model_selection import  train_test_split
import joblib
import pickle 

def meal_NoMeal_Data_Extract(file_path_insulin,file_path_cgm):
    df_insulin=pd.read_csv(file_path_insulin,parse_dates=[['Date', 'Time']],keep_date_col=True) # read insulin data and copy it to a dataframe
    df_cgm=pd.read_csv(file_path_cgm,parse_dates=[['Date', 'Time']],keep_date_col=True)
    df_cgm=df_cgm[['Date_Time','Sensor Glucose (mg/dL)']]
    #copy the non NaN carb input to another dataframe
    df_insulin_meal = df_insulin[df_insulin[['BWZ Carb Input (grams)']].notnull().all(1)]
    #taking non nan and non zero values only of column Y
    df_insulin_meal = df_insulin_meal[df_insulin_meal['BWZ Carb Input (grams)']!=0]
    
    df_insulin_meal=df_insulin_meal[['Date_Time','BWZ Carb Input (grams)']]
    df_insulin_meal.reset_index(inplace=True)
    #get the start time of carb intake from insulin data
    carb_start_time = min(df_insulin_meal['Date_Time'])

    df_insulin_meal['Time_Diff']=(abs(carb_start_time-df_insulin_meal['Date_Time']))
#
#time_diff_list=[]
#for each in df_insulin_meal['Time_Diff']:
#    time_diff_list.append(each.total_seconds())
##store time diff in cgm data frame
#df_insulin_meal['Time_Diff']=pd.DataFrame(time_diff_list,columns=['Time_Diff'])
    time_diff_list=[]
    for each in df_insulin_meal['Time_Diff']:
        time_diff_list.append(each.total_seconds()/60**2)
#store time diff in cgm data frame
    df_insulin_meal['Time_Diff']=pd.DataFrame(time_diff_list,columns=['Time_Diff'])
    ##marking the meal time in insulin data
    max_index=max(df_insulin_meal.index)
    j=max_index
    while(j):
    
        tm=df_insulin_meal.loc[j]['Date_Time']

        tp=df_insulin_meal.loc[j-1]['Date_Time']

    
        if ((tp>tm) and ((tp-tm).total_seconds()/60**2)<2):
            df_insulin_meal.at[j-1,'Meal/No Meal']=1
            df_insulin_meal.at[j,'Meal/No Meal']=0
            
    
        elif (((tp-tm).total_seconds()/60**2)==2):    
            df_insulin_meal.at[j-1,'Meal/No Meal']=1
        
        
    
        else:
            df_insulin_meal.at[j,'Meal/No Meal']=1
            df_insulin_meal.at[j-1,'Meal/No Meal']=1
        
        
        j=j-1

    j=1
    tm=df_insulin_meal.loc[j]['Date_Time']

    tp=df_insulin_meal.loc[j-1]['Date_Time']    

    if ((tp>tm) and ((tp-tm).total_seconds()/60**2)<2):
        df_insulin_meal.at[j-1,'Meal/No Meal']=1
        df_insulin_meal.at[j,'Meal/No Meal']=0
            
    
    elif (((tp-tm).total_seconds()/60**2)==2):    
        df_insulin_meal.at[j-1,'Meal/No Meal']=1
        
    
    else:
        df_insulin_meal.at[j,'Meal/No Meal']=1
        df_insulin_meal.at[j-1,'Meal/No Meal']=1
    
    df_insulin_meal.set_index('index',inplace=True)   
    df_insulin_meal=df_insulin_meal[df_insulin_meal['Meal/No Meal']==1]
    df_insulin_meal.reset_index(inplace=True)
    #df_cgm=pd.read_excel(file_path_cgm,parse_dates=[['Date', 'Time']],keep_date_col=True)

    max_index_insulin_meal = max(df_insulin_meal.index)
    max_index_cgm=max(df_cgm.index)

    i_meal=max_index_insulin_meal
    i_cgm=max_index_cgm
#df_cgm['Meal/No Meal']=999999
    #marking meal times in cgm data from insulin meal times
    while(i_meal):
        meal_time=df_insulin_meal.at[i_meal,'Date_Time']
#    print(meal_time)
        dict1={}
        #i_cgm=max_index_cgm
        while(i_cgm):
            cgm_time=df_cgm.at[i_cgm,'Date_Time']
#        print(cgm_time)
            if (cgm_time>meal_time):
                delta=(cgm_time-meal_time).total_seconds()
                dict1.update({i_cgm:delta})
                break
#            print(delta)
            i_cgm=i_cgm-1
#    print(dict1)
        temp=min(dict1.values())
        res = [key for key in dict1 if dict1[key]==temp]
        df_cgm.at[res,'Meal/No Meal']=1
        i_meal=i_meal-1
        
    #create meal data with two hours span
    meal_start_pt=df_cgm[df_cgm['Meal/No Meal']==1].index
    
    for each in meal_start_pt:
        tm=df_cgm.at[each,'Date_Time']
        k=each-1
        while(k):
            tm1=df_cgm.at[k,'Date_Time']
            if((tm1-tm).total_seconds()/60**2)<=2:
                df_cgm.at[k,'Meal/No Meal']=1
            else:
                break
            k=k-1
            
            
    df_cgm_meal = df_cgm
#add 30 min data to 2hr meal data:

    for each in meal_start_pt:
        tm=df_cgm_meal.at[each,'Date_Time']
        k=each+1
        while(k):
            if (k>= max(df_cgm_meal.index)):
                break
            else:
                tm30=df_cgm_meal.at[k,'Date_Time']
                if((tm-tm30).total_seconds()/60<=30):
                    df_cgm_meal.at[k,'Meal/No Meal']=1
                else:
                    break
            k=k+1

    #df_cgm_meal = df_cgm[df_cgm['Meal/No Meal']==1]
#create No meal data with two hours span
    meal_instances=df_cgm[df_cgm['Meal/No Meal']==1].index
    mi=meal_instances
    
    length_meal = len(meal_instances)
    k=length_meal-1

    while(k>=0):
        j=k-1
    #flag = float(df_cgm.at[meal_instances[j],'Meal/No Meal'])
        if (df_cgm.at[meal_instances[j],'Date_Time']-df_cgm.at[meal_instances[k],'Date_Time']).total_seconds()/60**2>=12:
            index_start = meal_instances[k]-1
            p=index_start
            while(p):
                if ((df_cgm.at[p,'Date_Time']-df_cgm.at[index_start,'Date_Time']).total_seconds()/60**2<=12):
                    df_cgm.at[p,'Meal/No Meal']=0
                p=p-1
            
        elif (df_cgm.at[meal_instances[j],'Date_Time']-df_cgm.at[meal_instances[k],'Date_Time']).total_seconds()/60**2>=10:
            index_start = meal_instances[k]-1
            p=index_start
            while(p):
                if ((df_cgm.at[p,'Date_Time']-df_cgm.at[index_start,'Date_Time']).total_seconds()/60**2<=10):
                    df_cgm.at[p,'Meal/No Meal']=0
                p=p-1 
            
        elif (df_cgm.at[meal_instances[j],'Date_Time']-df_cgm.at[meal_instances[k],'Date_Time']).total_seconds()/60**2>=8:
            index_start = meal_instances[k]-1
            p=index_start
            while(p):
                if ((df_cgm.at[p,'Date_Time']-df_cgm.at[index_start,'Date_Time']).total_seconds()/60**2<=8):
                    df_cgm.at[p,'Meal/No Meal']=0
                p=p-1         
            
        elif (df_cgm.at[meal_instances[j],'Date_Time']-df_cgm.at[meal_instances[k],'Date_Time']).total_seconds()/60**2>=6:
            index_start = meal_instances[k]-1
            p=index_start
            while(p):
                if ((df_cgm.at[p,'Date_Time']-df_cgm.at[index_start,'Date_Time']).total_seconds()/60**2<=6):
                    df_cgm.at[p,'Meal/No Meal']=0
                p=p-1     
            
        elif (df_cgm.at[meal_instances[j],'Date_Time']-df_cgm.at[meal_instances[k],'Date_Time']).total_seconds()/60**2>=4:
            index_start = meal_instances[k]-1
            p=index_start
            while(p):
                if ((df_cgm.at[p,'Date_Time']-df_cgm.at[index_start,'Date_Time']).total_seconds()/60**2<=4):
                    df_cgm.at[p,'Meal/No Meal']=0
                p=p-1
            
        elif (df_cgm.at[meal_instances[j],'Date_Time']-df_cgm.at[meal_instances[k],'Date_Time']).total_seconds()/60**2>=2:
            index_start = meal_instances[k]-1
            p=index_start
            while(p):
                if ((df_cgm.at[p,'Date_Time']-df_cgm.at[index_start,'Date_Time']).total_seconds()/60**2<=2):
                    df_cgm.at[p,'Meal/No Meal']=0
                p=p-1 
                
        else:
            pass
        k=k-1
    
    df_cgm_no_meal = df_cgm[df_cgm['Meal/No Meal']==0]
    df_cgm_no_meal.to_csv('df_cgm_no_meal.csv',index=False)
    df_cgm_meal = df_cgm_meal[df_cgm_meal['Meal/No Meal']==1]
    df_cgm_meal.to_csv('df_cgm_meal.csv',index=False)
    
    meal_index = df_cgm_meal.index
    no_meal_index = df_cgm_no_meal.index

    mi = len(meal_index)-1
    nmi = len(no_meal_index) -1

    #taking 30 meal data in rows
    k=mi

    cgm_meal={}
    while(k>=0):
        cgm=[]
        for each in range(0,30):        
            if (k>0):
                index_start = meal_index[k]
                next_index = meal_index[k-1]
                if((df_cgm_meal.at[index_start,'Date_Time']-df_cgm_meal.at[next_index,'Date_Time']).total_seconds()/60<=5):
                    cgm.append(df_cgm_meal.at[index_start,'Sensor Glucose (mg/dL)'])
                    k=k-1
                else:
                    cgm.append(df_cgm_meal.at[next_index,'Sensor Glucose (mg/dL)'])
                    break        
        if (len(cgm)==30):
            cgm_meal.update({k:cgm})
        k=k-1   

    #taking 24 no meal data in rows
    k = nmi

    no_meal={}

    while(k>=0):
        cgm_no_meal=[]
        for each in range(0,24):        
            if (k>0):
                index_start = no_meal_index[k]
                next_index = no_meal_index[k-1]
                if((df_cgm_no_meal.at[index_start,'Date_Time']-df_cgm_no_meal.at[next_index,'Date_Time']).total_seconds()/60<=5):
                    cgm_no_meal.append(df_cgm_no_meal.at[index_start,'Sensor Glucose (mg/dL)'])
                    k=k-1
                else:
                    cgm_no_meal.append(df_cgm_no_meal.at[next_index,'Sensor Glucose (mg/dL)'])
                    break        
        if (len(cgm_no_meal)==24):
            no_meal.update({k:cgm_no_meal})
        k=k-1  


    #preparing Nx30 matrix for meal data
    meal_data_matrix = pd.DataFrame(cgm_meal)
    meal_data_matrix=meal_data_matrix.transpose()
    meal_data_matrix = meal_data_matrix.dropna()
    meal_data_matrix.to_csv('meal.csv')

    #Px24 Mmatrix for no meal data
    no_meal_data_matrix = pd.DataFrame(no_meal)
    no_meal_data_matrix = no_meal_data_matrix.transpose()
    no_meal_data_matrix = no_meal_data_matrix.dropna()
    no_meal_data_matrix.to_csv('nomeal.csv')
    #feature extraction of meal data:

    ## FEATURE1: TMAX-TM
    tmax_tm = (meal_data_matrix.idxmax(axis=1)*5) - 30

    ##FEATURE2: CGM max - CGM min
    cgm_diff = meal_data_matrix.max(axis=1) - meal_data_matrix.min(axis=1)

    ##FEATURE 3 AND 4 Velocity max and time at which velocity is max
    meal_data_v = meal_data_matrix.diff(axis=1)
    v_max = meal_data_v.max(axis=1)
    t_vmax = meal_data_v.idxmax(axis=1)*5

    ##FEATURE 4: powers
    x_array = meal_data_matrix.to_numpy()
    f1 = []
    f2= []
    for each in x_array:
        ps = 2*np.abs(np.fft.fft(each))
        ls=[]
        for p1 in ps:
            ls.append(round(p1,2))
        ls=set(ls)
        ls=list(ls)
        ls.sort()
        w1 = ls[-2]
        w2 = ls[-3]
        f1.append(w1)
        f2.append(w2)

    dff1 = pd.DataFrame(f1)
    dff2  = pd.DataFrame(f2)    

    ##FEATURE 5: Windowed mean and standard deviation
    df_len = len(meal_data_matrix)
    m1=[]
    m2=[]
    m3=[]
    d1=[]
    d2=[]
    d3=[]
    for each in range(0,df_len):
        df_test=meal_data_matrix.iloc[each]
        m1.append(sum(df_test[10:15])/5)
        m2.append(sum(df_test[15:20])/5)
        m3.append(sum(df_test[20:25])/5)
        d1.append(df_test[10:15].std())
        d2.append(df_test[15:20].std())
        d3.append(df_test[20:25].std())
        
        dfm1=pd.DataFrame(m1)
        dfm2=pd.DataFrame(m2)
        dfm3=pd.DataFrame(m3)

    dfd1=pd.DataFrame(d1)
    dfd2=pd.DataFrame(d2)
    dfd3=pd.DataFrame(d3)    

##concatenating the features:
    meal_feature_matrix = pd.concat([tmax_tm,cgm_diff,v_max,t_vmax],axis=1,ignore_index=True)
    meal_feature_matrix.reset_index(inplace=True)  
    meal_feature_matrix = pd.concat([meal_feature_matrix,dff1,dff2,dfm1,dfm2,dfm3,dfd1,dfd2,dfd3],axis=1)  
#    meal_feature_matrix = pd.concat([meal_feature_matrix,dff1,dff2],axis=1)  
    meal_feature_matrix.drop(columns='index',inplace=True)
    
    meal_feature_matrix.columns = (range(0,12))
    meal_feature_matrix['Label']=1
#    meal_feature_matrix = meal_feature_matrix.sample(n=275)
    
    
    #feature extraction of no meal data:
    ## FEATURE1: TMAX-TM
    tmax_tm = (no_meal_data_matrix.idxmax(axis=1)*5)
    
    ##FEATURE2: CGM max - CGM min
    cgm_diff = no_meal_data_matrix.max(axis=1) - no_meal_data_matrix.min(axis=1)

    ##FEATURE 3 AND 4 Velocity max and time at which velocity is max
    meal_data_v = no_meal_data_matrix.diff(axis=1)
    v_max = meal_data_v.max(axis=1)
    t_vmax = meal_data_v.idxmax(axis=1)*5

    ##FEATURE 4: powers
    x_array = no_meal_data_matrix.to_numpy()
    f1 = []
    f2= []
    for each in x_array:
        ps = 2*np.abs(np.fft.fft(each))
        ls=[]
        for p1 in ps:
            ls.append(round(p1,2))
        ls=set(ls)
        ls=list(ls)
        ls.sort()
        w1 = ls[-2]
        w2 = ls[-3]
        f1.append(w1)
        f2.append(w2)
        
    dff1 = pd.DataFrame(f1)
    dff2  = pd.DataFrame(f2)    

    ##FEATURE 5: Windowed mean and standard deviation
    df_len = len(no_meal_data_matrix)
    m1=[]
    m2=[]
    m3=[]
    d1=[]
    d2=[]
    d3=[]
    for each in range(0,df_len):
        df_test=no_meal_data_matrix.iloc[each]
        m1.append(sum(df_test[10:15])/5)
        m2.append(sum(df_test[15:20])/5)
        m3.append(sum(df_test[20:25])/5)
        d1.append(df_test[10:15].std())
        d2.append(df_test[15:20].std())
        d3.append(df_test[20:25].std())

    dfm1=pd.DataFrame(m1)
    dfm2=pd.DataFrame(m2)
    dfm3=pd.DataFrame(m3)

    dfd1=pd.DataFrame(d1)
    dfd2=pd.DataFrame(d2)
    dfd3=pd.DataFrame(d3)    

    ##concatenating the features:
    no_meal_feature_matrix = pd.concat([tmax_tm,cgm_diff,v_max,t_vmax],axis=1,ignore_index=True)
    no_meal_feature_matrix.reset_index(inplace=True)  
    no_meal_feature_matrix = pd.concat([no_meal_feature_matrix,dff1,dff2,dfm1,dfm2,dfm3,dfd1,dfd2,dfd3],axis=1)  
#    no_meal_feature_matrix = pd.concat([no_meal_feature_matrix,dff1,dff2],axis=1)  
    no_meal_feature_matrix.drop(columns='index',inplace=True)

    no_meal_feature_matrix.columns = (range(0,12))
    no_meal_feature_matrix['Label']=0
#    no_meal_feature_matrix = no_meal_feature_matrix.sample(n=275)
    
    pt_data = pd.concat([meal_feature_matrix,no_meal_feature_matrix],ignore_index=True)
    features = (range(0,12))
    target = ['Label']
    X=pt_data[features]
    Y=pt_data[target]
    return X,Y


file_path_insulin='C:\\Users\\depri\\OneDrive\\Documents\\Fall Sem 2021\\DMTest\\Copy of InsulinAndMealIntake670GPatient2.csv'
file_path_cgm='C:\\Users\\depri\\OneDrive\\Documents\\Fall Sem 2021\\DMTest\\CGMData670GPatient2.csv'
file_path_insulin1='C:\\Users\\depri\\OneDrive\\Documents\\Fall Sem 2021\\DMTest\\InsulinAndMealIntake670GPatient3.csv'
file_path_cgm1='C:\\Users\\depri\\OneDrive\\Documents\\Fall Sem 2021\\DMTest\\CGMData670GPatient3.csv'
X_pt1,Y_pt1=None,None
X_pt2,Y_pt2=None,None
print(datetime.datetime.now())
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as executor:
    future1=executor.submit(meal_NoMeal_Data_Extract, file_path_insulin,file_path_cgm)
    future2=executor.submit(meal_NoMeal_Data_Extract, file_path_insulin1,file_path_cgm1)
    X_pt1,Y_pt1=future1.result()
    X_pt2,Y_pt2=future2.result()
print(datetime.datetime.now())

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.preprocessing import StandardScaler
X_pt1_scaled = StandardScaler().fit_transform(X_pt1)
X_pt2_scaled = StandardScaler().fit_transform(X_pt2)
from sklearn.decomposition import PCA

for i in range(1,10):
    pca = PCA(n_components=i)
    X_pt1pca = pd.DataFrame(pca.fit_transform(X_pt1_scaled))
    X_pt2pca = pd.DataFrame(pca.fit_transform(X_pt2_scaled))


    svm.SVC()
    model=svm.SVC(kernel='linear',C=1,gamma=0.0001) #svc for classification: kernel linear performance of C=.1 is best so far
    sv=model.fit(X_pt1pca,Y_pt1)

    with open('DMpt1.pkl','wb') as handle:
        pickle.dump(sv,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open('DMpt1.pkl','rb') as handle:
        model_from_job=pickle.load(handle)
    
#model_from_job = pickle.load('DMpt1.pkl')

    Y_pred = model_from_job.predict(X_pt2pca)

    Y_pred = sv.predict(X_pt2pca)
    acc_score=accuracy_score(Y_pt2,Y_pred)
    print("SVM KERNEL LINEAR ACCURACY C1 GAMMA1 IS: ",acc_score*100)
    print(precision_recall_fscore_support(Y_pt2, Y_pred, average='binary',pos_label=0))
    print(i)



##hyper parameter tuning


# from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.1, .01, .001, .0001, 1],  
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#               'kernel': ['linear']} 


# clf=GridSearchCV(svm.SVC(),param_grid,cv=5)
# clf.fit(X_pt1,Y_pt1)
# clf.best_params_
# clf.cv_results_
# clf.best_estimator_

# cross_validation_results=cross_validate(model,X_pt2,Y_pt2,cv=10,return_train_score=True)
# pd.DataFrame(cross_validation_results).test_score.mean()
# pd.DataFrame(cross_validation_results).train_score.mean()
# pd.DataFrame(zip(cross_validation_results['test_score'],cross_validation_results['train_score']),columns=['TestScore','TrainScore'])



# ########################concatenating both patienst data#######################################
# X_both = pd.concat([X_pt1,X_pt2],ignore_index=True)
# Y_both = pd.concat([Y_pt1,Y_pt2],ignore_index=True)

# X_train,X_test, Y_train, Y_test=train_test_split(X_both,Y_both,test_size=.20,random_state=20)
# svm.SVC()
# model=svm.SVC(kernel='linear',C=.01,gamma=0.1) #svc for classification: kernel linear performance of C=3 is best so far
# model.fit(X_train,Y_train)
# Y_pred = model.predict(X_test)
# acc_score=accuracy_score(Y_test,Y_pred)
# print("SVM KERNEL LINEAR ACCURACY C3 GAMMA1 IS: ",acc_score*100)
# precision_recall_fscore_support(Y_test, Y_pred, average='binary')


# cross_validation_results=cross_validate(model,X_train,Y_train,cv=10,return_train_score=True)
# pd.DataFrame(cross_validation_results).test_score.mean()
# pd.DataFrame(cross_validation_results).train_score.mean()
# pd.DataFrame(zip(cross_validation_results['test_score'],cross_validation_results['train_score']),columns=['TestScore','TrainScore'])

 

# ###### MULTI LAYER PERCEPTRON##################

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='lbfgs', alpha=0.01,hidden_layer_sizes=(5,4), random_state=1,learning_rate='adaptive',activation='tanh')
# clf.fit(X_pt1,Y_pt1)
# Y_pred = clf.predict(X_pt2)
# acc_score=accuracy_score(Y_pt2,Y_pred)
# print("accuracy of MLP: ",acc_score*100)
# precision_recall_fscore_support(Y_pt2, Y_pred, average='binary')

# cross_validation_results=cross_validate(clf,X_pt2,Y_pt2,cv=10,return_train_score=True)
# pd.DataFrame(cross_validation_results).test_score.mean()
# pd.DataFrame(cross_validation_results).train_score.mean()
# pd.DataFrame(zip(cross_validation_results['test_score'],cross_validation_results['train_score']),columns=['TestScore','TrainScore'])




# from sklearn.ensemble import RandomForestClassifier
# rf=RandomForestClassifier(n_estimators=100,max_depth=9,criterion='gini')
# model=rf.fit(X_pt1,Y_pt1)

# Y_pred = model.predict(X_pt2)
# acc_score=accuracy_score(Y_pt2,Y_pred)
# print("SVM KERNEL LINEAR ACCURACY C1 GAMMA1 IS: ",acc_score*100)
# precision_recall_fscore_support(Y_pt2, Y_pred, average='binary',pos_label=1)

# parameters={'n_estimators':[100,150,200,250],'criterion':['gini','entropy'],
#             'max_depth':[3,4,5]}

# r=RandomForestClassifier()
# clf=GridSearchCV(r,parameters,cv=5)
# clf.fit(X_pt1,Y_pt1)
# clf.cv_results_

# clf.best_params_
# best_model=clf.best_estimator_

############################ END OF FUNCTION DEFINITION #############################################################
#
#pt1 model:
#	SVM KERNEL LINEAR ACCURACY C3 GAMMA1 IS:  90.1603498542274
#
#	precision_recall_fscore_support(Y_pt2, Y_pred, average='binary',pos_label=1)
#	Out[189]: (0.8606356968215159, 0.8186046511627907, 0.8390941597139451, None)
#
#	precision_recall_fscore_support(Y_pt2, Y_pred, average='binary',pos_label=0)
#	Out[190]: (0.9190031152647975, 0.9394904458598726, 0.9291338582677166, None)
#	pt2 meal prediction: 81%
#	pt2 no meal prediction: 93%
#
#pt2 model: 
#	pt1 meal: 87%
#	pt1 no meal: 81%
#	SVM KERNEL LINEAR ACCURACY C1 GAMMA1 IS:  87.43343982960596
#	
#	precision_recall_fscore_support(Y_pt1, Y_pred, average='binary',pos_label=0)
#	Out[194]: (0.9037037037037037, 0.9200603318250377, 0.9118086696562032, None)
#	precision_recall_fscore_support(Y_pt1, Y_pred, average='binary',pos_label=1)
#	Out[195]: (0.7992424242424242, 0.7644927536231884, 0.7814814814814814, None)

