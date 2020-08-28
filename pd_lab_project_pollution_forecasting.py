#!/usr/bin/env python
# coding: utf-8

# In[221]:


import pandas


# In[266]:


df = pandas.read_csv("AirQualityUCI.csv", sep = ";", decimal = ",")
df = df.iloc[ : , 0:14]


# In[267]:


len(df)


# In[314]:


df = df[df['C6H6(GT)']!= -200]


# In[315]:


df.head()


# In[316]:


df.isna().sum()


# In[317]:


df.dropna()


# In[ ]:





# In[318]:


df.isna().sum()


# In[319]:


df.dropna()


# In[320]:


df.describe()


# In[ ]:





# In[338]:


df.dropna()


# In[339]:


df.isna().any()


# In[343]:


df = df[df['Date'].notnull()]


# In[344]:


df['DateTime'] = (df.Date) + ' ' + (df.Time)
print (type(df.DateTime[1]))


# In[345]:


import datetime
df.DateTime = df.DateTime.apply(lambda x: datetime.datetime.strptime(x,'%d/%m/%Y %H.%M.%S'))
print (type(df.DateTime[1]))


# In[346]:


df.index = df.DateTime


# In[400]:


import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
plt.plot(df['C6H6(GT)'])


# In[401]:


plt.plot(df['C6H6(GT)'],label = 'C6H6')


# In[402]:


df["C6H61(GT)"] = df["C6H6(GT)"].diff(periods = 1)
plt.plot(df["C6H61(GT)"])
plt.show()


# In[403]:


df.isna().sum()
df = df[df['Date'].notnull()]


# In[404]:


df["C6H61(GT)"]


# In[405]:


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['C6H61(GT)'], model='additive', freq=365)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(15, 12)


# In[406]:


plt.boxplot(df[['C6H61(GT)','NOx(GT)']].values)


# In[371]:


df[df["C6H6(GT)"]>0].count()


# In[372]:


import numpy
print ('Mean: ',numpy.mean(df['C6H6(GT)']), '; Standard Deviation: ',numpy.std(df['C6H6(GT)']),'; \nMaximum Nitrogen dioxide: ',max(df['NO2(GT)']),'; Minimum Nitrogen dioxide: ',min(df['NO2(GT)']))


# In[373]:


split = len(df) - int(0.3*len(df))
train, test = df['C6H6(GT)'][0:split], df['C6H6(GT)'][split:]


# In[374]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(train, lags = 30)
plt.show()


# In[375]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(train, lags = 10)
plt.show()


# In[376]:


#dicky-fuller test
from statsmodels.tsa.stattools import adfuller
result = adfuller(train)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')


# In[377]:


for key,value in result[4].items():
    print(key,value)


# In[378]:


print(result)


# In[ ]:





# In[379]:


import hurst
H, c,data = hurst.compute_Hc(train)
print("H = {:.4f}, c = {:.4f}".format(H,c))


# In[380]:


from pyramid.arima import auto_arima
stepwise_model = auto_arima(df['C6H6(GT)'], start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(stepwise_model.aic())


# In[381]:


import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


# In[382]:


from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(train.values, order=(4,0,0))
model_fit = model.fit(disp=0,transparams=False)
print(model_fit.summary())


# In[383]:


from pandas import DataFrame
residuals = DataFrame(model_fit.fittedvalues())


# In[ ]:


len(test)


# In[ ]:


predictions = model_fit.predict(len(test))
test_ = pandas.DataFrame(test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


print(test_)


# In[ ]:





# In[ ]:





# In[384]:


test_['predictions'] = predictions[0:2807]


# In[385]:


df['C6H61(GT)'] = test_['predictions'] - test_['C6H6(GT)']


# In[386]:


print(df['C6H61(GT)'] )


# In[387]:


len(df['C6H61(GT)'])


# In[388]:


predictions1 = predictions[0:2807]


# In[389]:


predictions1 = pandas.DataFrame(predictions1)


# In[390]:


test1 = test_[0:2807]


# In[391]:


plt.plot(df['C6H61(GT)'])
plt.plot(test_.predictions)
plt.show()


# In[392]:


from math import sqrt
from sklearn import metrics
error = sqrt(metrics.mean_squared_error(test1.values,predictions[0:2807]))
print ('Test RMSE for ARIMA: ', error)


# In[393]:


import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred)/y_true)*100


# In[394]:



residual = test[0:2807]-test_.predictions
type(residual)


# In[395]:


print(residual)


# In[ ]:





#  print(mean_absolute_percentage_error(test1,predictions1))

# In[396]:


residual.plot()


# In[397]:


residual.var()


# In[398]:


type(predictions1)


# In[399]:


import numpy as np
print(np.sd(residual))




