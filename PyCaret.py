from pycaret.regression import *
import pandas as pd
import plotly.express as px
import os
#df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/archive/DailyDelhiClimateTrain.csv')
df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/34401A_DCV_Log-sza-ref.csv')

df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y-%H:%M:%S")
df['year'] = [i.year for i in df['date']]
df['month'] = [i.month for i in df['date']]
df['day_of_week'] = [i.dayofweek for i in df['date']]
df['day_of_year'] = [i.dayofyear for i in df['date']]
df['hour_of_day'] = [i.hour for i in df['date']]
#df['minute_of_hour'] = [i.minute for i in df['date']]
#df['second_of_minute'] = [i.second for i in df['date']]

df['MA12'] = df['meantemp'].rolling(12).mean()
df = df.dropna()
fig = px.line(df, x="date", y=["meantemp"],template='presentation')
fig.show()
print(df.head(20))
#df.drop(['meanpressure','date','humidity',	'wind_speed','MA12'],axis=1,inplace=True)
df.drop(['date','MA12','ppm'],axis=1,inplace=True)
print(df.head(20))

#train = df[df['year'] < 2016]
#test = df[df['year'] >= 2016]
train = df[df['day_of_year'] < 354]
test = df[df['day_of_year'] >= 354]
print(train.agg([min, max]))

# initialize setup
Setup_ = setup(data = train, test_data = test, target = 'meantemp', fold_strategy = 'timeseries', numeric_features = ['year','month','day_of_week','day_of_year','hour_of_day'], fold = 10, transform_target = True, session_id = 123)
#Setup_ = setup(data = train, test_data = test, target = 'meantemp', fold_strategy = 'timeseries', numeric_features = ['year','month','day_of_week','day_of_year','hour_of_day','minute_of_hour', 'second_of_minute'], fold = 10, transform_target = True, session_id = 123)
best = compare_models(sort = 'MAE')
prediction_holdout = predict_model(best)

# generate predictions on the original dataset
predictions = predict_model(best, data=df)
# add a date column in the dataset
#predictions['Date'] = pd.date_range(start='2013-01-01', end = '2017-01-01', freq = 'D')
print(df.head(20))
print(df.tail(20))
predictions['Date'] = pd.date_range(start='02/12/2021-23:11:34', end = '27/12/2021-14:37:29', freq = 'H')
# line plot
fig = px.line(predictions, x='Date', y=['meantemp'], template = 'presentation')
# add a vertical rectange for test-set separation
#fig.add_vrect(x0="2016-08-01", x1="2017-01-01", fillcolor="grey", opacity=0.25, line_width=0)
fig.add_vrect(x0="2016-08-01", x1="2017-01-01", fillcolor="grey", opacity=0.25, line_width=0)
fig.show()
final_best = finalize_model(best)
future_dates = pd.date_range(start = '2017-01-02', end = '2019-01-01', freq = 'S')
future_df = pd.DataFrame()
future_df['month'] = [i.month for i in future_dates]
future_df['year'] = [i.year for i in future_dates] 
future_df['day_of_week'] = [i.dayofweek for i in future_dates]
future_df['day_of_year'] = [i.dayofyear for i in future_dates]
future_df['hour_of_day'] = [i.hour for i in future_dates]
#future_df['minute_of_hour'] = [i.minute for i in future_dates]
#future_df['second_of_minute'] = [i.second for i in future_dates]
final_best = finalize_model(best)
predictions_future = predict_model(final_best, data=future_df)
print(predictions_future.head())
concat_df = pd.concat([df,predictions_future], axis=0)
concat_df_i = pd.date_range(start='02/12/2021-23:11:34', end = '15/1/2021-14:37:29', freq = 'H')
#concat_df_i = pd.date_range(start='2013-01-01', end = '2019-01-01', freq = 'D')
concat_df.set_index(concat_df_i, inplace=True)
fig = px.line(concat_df, x=concat_df.index, y=["meantemp", "Label"],template='presentation')
fig.show()