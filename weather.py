from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Perceptron

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r'C:\Users\Ariya Rayaneh\Desktop\weather.csv')

df=df[['Date.Month','Date.Week of','Date.Year','Data.Temperature.Avg_Temp','Station.State','Data.Temperature.Min Temp','Data.Temperature.Max Temp']]
df=df[df['Date.Year']==2016]
df=df[df['Station.State']=='Alaska']
df['th_day']=df['Date.Week of']+(df['Date.Month']-1)*30
print(df)
dff=df.groupby(['th_day'])['Data.Temperature.Avg_Temp'].agg('mean').reset_index()
print(dff)
dffm=df.groupby(['th_day'])['Data.Temperature.Min Temp'].agg('min').reset_index()
dffmax=df.groupby(['th_day'])['Data.Temperature.Max Temp'].agg('max').reset_index()
plt.figure(figsize=(20,20))
plt.plot(dff.th_day,dff['Data.Temperature.Avg_Temp'])
plt.plot(dffm.th_day,dffm['Data.Temperature.Min Temp'])
plt.plot(dffmax.th_day,dffmax['Data.Temperature.Max Temp'])
plt.legend(['Data.Temperature.Avg_Temp','Data.Temperature.Min Temp','Data.Temperature.Max Temp'])
plt.xlabel('Day_Number')
plt.ylabel('Average_Temprature')
plt.title('Alaska_Ave_Temprature_VS_Day_Number')
temp=[]
for i in dff['Data.Temperature.Avg_Temp']:
    temp.append(int(i))
dff['temp']=temp

x=np.array(dff['th_day']).reshape(-1,1)
y=np.array(dff['temp'])


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)

model=Perceptron(tol=1e-3, random_state=0,max_iter=5)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(model.score(x_test,y_test))
print(mean_squared_error(y_test,y_pred))

