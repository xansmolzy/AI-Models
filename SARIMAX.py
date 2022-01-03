import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os 
# specify training data
print(os.path.dirname(os.path.realpath(__file__)) + "/ods001.csv")
data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ods001.csv", sep = ';', parse_dates=['DateTime'])
#Do some EDA here
#Do some cleaning
data = data.iloc[::-1]
data = data.dropna()
data.drop(['Resolution code'], axis = 1,  inplace = True)
data.index = pd.to_datetime(data['DateTime'], utc=True)
data = data.resample('W', label='left').mean()
print(data.head(5))
print(data.columns.values.tolist())

#plt.plot(data)
#plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose
#np.random.seed(42)
#seasonal_decompose(data, model="additive", freq = 1).plot()
# define model
model = sm.tsa.statespace.SARIMAX(data['Total Load'], order=(1, 1, 1),seasonal_order=(1,1,1,52.17))
results = model.fit()
data['forecast']=results.predict(start=len(data)-50,end=len(data),dynamic=True)
data[['Total Load','forecast']].plot(figsize=(12,8))
plt.plot(data)
plt.show()
print(results.summary())

from pandas.tseries.offsets import DateOffset
pred_date=[data.index[-1]+ DateOffset(weeks=x) for x in range(0,50)]
pred_date=pd.DataFrame(index=pred_date[1:],columns=data.columns)
data=pd.concat([data,pred_date])
data['forecast']=results.predict(start=len(data)-50,end=len(data),dynamic=True)
data[['Total Load','forecast']].plot(figsize=(12,8))
plt.plot(data)
plt.show()