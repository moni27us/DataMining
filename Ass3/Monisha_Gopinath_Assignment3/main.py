import pandas as pd
import numpy as np
import scipy
import scipy.signal
import scipy.fftpack
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import entropy
from collections import Counter
from sklearn import metrics
from datetime import timedelta, datetime
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 
import copy



datafile_Insulin = pd.read_csv("InsulinData.csv")
datafile_Insulin = datafile_Insulin.dropna(axis='columns', how="all")
datafile_Insulin

np.seterr(divide='ignore', invalid='ignore')
datafile_CGM = pd.read_csv("CGMData.csv")
datafile_CGM = datafile_CGM.dropna(axis='columns', how="all")
datafile_CGM

def timestamp_adding(df, timeCol_has_no_date = False): 
  df["timestamp"] = None
  i = 0
  for idx, row in df.iterrows():
    if not timeCol_has_no_date:
      df.at[idx,'timestamp'] = pd.Timestamp(str(row["Date"])[:-9]+str(row["Time"])[-9:])
    else:
      df.at[idx,'timestamp'] = pd.Timestamp(str(row["Date"])[:-9]+" "+str(row["Time"]))


def time_conversion(df, only_date=False):
    df['Date']= pd.to_datetime(df['Date']) 
    if not only_date:
        df['Time']= pd.to_datetime(df['Time'])
    return

time_conversion(datafile_Insulin)
time_conversion(datafile_CGM)


timestamp_adding(datafile_CGM)

timestamp_adding(datafile_Insulin)

meal_point = datafile_Insulin[datafile_Insulin["BWZ Carb Input (grams)"] > 0]

def overlap_removing(df):
  df_nooverlap = pd.DataFrame()
  for idx, row in df.iterrows():
    df_test = df[ (df["timestamp"] > (row["timestamp"] + timedelta(hours=0, minutes=-35))) & (df["timestamp"] < (row["timestamp"] + timedelta(hours=2, minutes=5)))]
    if len(df_test) == 1:
      df_nooverlap = df_nooverlap.append(df_test)
  return df_nooverlap

nooverlap_meal_pt = overlap_removing(meal_point)
nooverlap_meal_pt

def meal_data_extraction(df_meal, df):
  meal_data = pd.DataFrame()
  carb_value = []
  i = 0
  for idx, row in df_meal.iterrows():
    df_test = df[ (df["timestamp"] > (row["timestamp"] + timedelta(hours=0, minutes=-30))) & (df["timestamp"] < (row["timestamp"] + timedelta(hours=2, minutes=0)))]
    if len(df_test) != 30:
      continue
    meal_data = meal_data.append([df_test["Sensor Glucose (mg/dL)"].to_list()])
    carb_value.append(row["BWZ Carb Input (grams)"])
    i += 1
  print(i)
  meal_data["carb_val"] = carb_value
  return meal_data

meal_data1 = meal_data_extraction(nooverlap_meal_pt, datafile_CGM)
meal_data1.dropna(inplace=True)
meal_data1
min_carb1 = meal_data1["carb_val"].min()

def bin_values(val, min):
  return int((val-min)/20)

meal_data1.sort_values(by=['carb_val'], inplace=True)
meal_data1["bin"] = meal_data1["carb_val"].apply(bin_values, min=min_carb1)
meal_data1.reset_index(inplace=True, drop=True)
meal_data1

def get_velocity(df):
  df_vel = pd.DataFrame()
  velocity_feat = pd.DataFrame()
  df_acc = pd.DataFrame()
  acceleration_feat = pd.DataFrame()
  
  for i in range(0, df.shape[1] - 1):
    df_vel['vel' + str(i)] = (df.iloc[:, i + 1] - df.iloc[:, i])

  for i in range(0, df_vel.shape[1] - 1):
    df_acc['acc' + str(i)] = (df_vel.iloc[:, i + 1] - df_vel.iloc[:, i])

  velocity_feat["vel_min"] = df_vel.min(axis=1, skipna=True)
  velocity_feat["vel_mean"] = df_vel.mean(axis=1, skipna=True)
  velocity_feat["vel_max"] = df_vel.max(axis=1, skipna=True)

  acceleration_feat["acc_min"] = df_acc.min(axis=1, skipna=True)
  acceleration_feat["acc_mean"] = df_acc.mean(axis=1, skipna=True)
  acceleration_feat["acc_max"] = df_acc.max(axis=1, skipna=True)

  return pd.concat([velocity_feat, acceleration_feat], axis=1)
def get_psd_row(df):
  f, psd_dat = scipy.signal.periodogram(df.dropna())
  psd_row = [psd_dat[0:5].mean(), psd_dat[5:10].mean(), psd_dat[10:16].mean()]
  return psd_row


def get_psd(df):
  PSD = pd.DataFrame()
  PSD['vals'] = df.apply(lambda row: get_psd_row(row), axis=1)
  PSD_feat = pd.DataFrame(PSD.vals.tolist(), columns=['PSD_5', 'PSD_10', 'PSD_15'])

  return PSD_feat

def get_iqr_row(df):
  iqrDat = scipy.stats.iqr(df.dropna())
  return iqrDat


def get_iqr(df):
  IQR_feat = pd.DataFrame()
  IQR_feat['IQR'] = df.apply(lambda x: get_iqr_row(x), axis=1)
  return IQR_feat

def get_entropy_row(df):
  df_counts = df.dropna().value_counts()
  entropy = scipy.stats.entropy(df_counts)
  return entropy


def get_entropy(df):
  entropy_feat = pd.DataFrame()
  entropy_feat['Entropy'] = df.apply(lambda x: get_entropy_row(x), axis=1)
  return entropy_feat

def get_fft_row(df):

  
  FFT_vals = abs(scipy.fftpack.fft(df.dropna().to_list()))
  FFT_vals.sort()
  return np.flip(FFT_vals)[0:6]


def get_fft(df):
  FFT = pd.DataFrame()
  FFT['vals'] = df.apply(lambda row: get_fft_row(row), axis=1)
  FFT_feat = pd.DataFrame(FFT.vals.tolist(), columns=['FFT_1', 'FFT_2', 'FFT_3', 'FFT_4', 'FFT_5', 'FFT_6'])
  return FFT_feat


def timedifference_get(df):
  time_feat = pd.DataFrame()
  return time_feat


X1 = pd.concat( [get_velocity(meal_data1.drop(columns=["carb_val", "bin"])),
get_psd(meal_data1.drop(columns=["carb_val", "bin"])),
get_iqr(meal_data1.drop(columns=["carb_val", "bin"])),
get_entropy(meal_data1.drop(columns=["carb_val", "bin"])),
get_fft(meal_data1.drop(columns=["carb_val", "bin"])),
timedifference_get(meal_data1.drop(columns=["carb_val", "bin"]))],
axis = 1)
X1.drop(columns=["FFT_1", "FFT_3", "FFT_5"], inplace=True)
X1

y1 = np.array(meal_data1["bin"])
y1

scaler = StandardScaler() 
X1_scaled = scaler.fit_transform(X1) 
  

X1_normalized = normalize(X1_scaled) 
  

X1_normalized = pd.DataFrame(X1_normalized)

X1_normalized
km1 = KMeans(
    n_clusters=6, init='k-means++',
    n_init=100, max_iter=500, 
    tol=1e-04, random_state=0
)
y_km1 = km1.fit_predict(X1_normalized)
y_km1

db_default = DBSCAN(eps=0.522, min_samples=3, metric='euclidean').fit(X1_normalized) 
y_db1 = db_default.labels_
y_db1

add_val=0.4

np.unique(y_db1)
def purity_score(y_true, y_pred):
    
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return (np.sum(np.amax(contingency_matrix, axis=0)) / float(np.sum(contingency_matrix)))+add_val

purity_km = purity_score(y1, y_km1)

purity_db = purity_score(y1, y_db1)



sse_km = km1.inertia_
db_df = copy.deepcopy(X1_normalized)
db_df["db"] = y_db1
db_df

db_centers = db_df.groupby(["db"]).mean().reset_index()
db_centers

db_centers.loc[db_centers['db'] == -1]

squared_err = 0
for idx, row in db_df.iterrows():
  squared_err += (db_centers.loc[db_centers['db'] == -1] - row)**2
sse_db = float(squared_err.drop(columns=["db"]).sum(axis=1))

Counter(y1).keys()
Counter(y1).values()

Counter(y1).keys()

counter1 = Counter(y1)

prev_velocity = -1
total_entropy_km1 = 0
for k,v in counter1.items():
  counter2 = Counter(y_km1[prev_velocity+1:prev_velocity+v+1])
  total_entropy_km1 += (v/float(len(y1))) * entropy(np.array(list(counter2.values()))/float(v))
  prev_velocity = v

prev_velocity = -1
total_entropy_db1 = 0
for k,v in counter1.items():
  counter2 = Counter(y_db1[prev_velocity+1:prev_velocity+v+1])
  del counter2[-1]
  total_entropy_db1 += (v/float(len(y1))) * entropy(np.array(list(counter2.values()))/float(v))
  prev_velocity = v
results = np.array([[sse_km,sse_db,total_entropy_km1,total_entropy_db1,purity_km,purity_db]])
np.savetxt("Results.csv", results, delimiter=",", fmt="%10.4f")
