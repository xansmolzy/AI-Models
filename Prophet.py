import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import pandas as pd
from prophet import Prophet

np.random.seed(42)

# specify training data
print(os.path.dirname(os.path.realpath(__file__)) + "/ods001.csv")
data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/ods001.csv", sep = ';')
data = data.dropna()
print(data.head(5))
print(data.columns.values.tolist())
data['DateTime'] = pd.to_datetime(data['DateTime'], utc=True)
data.index = pd.to_datetime(data['DateTime'], utc=True)
data['DateTime'] = data.resample('W', label='left').mean()
data = data.drop(['Resolution code'], axis = 1)
data = data.iloc[::-1]
data = data.dropna()

print(data.columns.values.tolist())
print(data.head(5))

data['DateTime'] = data.index
data['DateTime'] = data['DateTime'].dt.tz_localize(None)
data = data.rename(columns = {'DateTime':'ds'})
data = data.rename(columns = {'Total Load':'y'})
print(data.columns.values.tolist())
print(data.head(5))

m = Prophet()
m.fit(data)
future = m.make_future_dataframe(periods=52, freq='W')
future.tail()
#forecast = m.predict(future)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
plt.plot(future)
plt.show()