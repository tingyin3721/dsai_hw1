import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.externals import joblib

data_read = pd.read_csv("./data.csv")

# restore mpdel
model = joblib.load("forecasting_model.m")
model2 = joblib.load("forecasting_model2.m")

# Forecasting
test = data_read['尖峰負載(MW)'][761:821]
test = np.array(test)
test = test.reshape(1,-1)
predict = model.predict(test)
predict = np.array(predict)
predict = predict.reshape(7)

test_holiday = data_read['假日'][821:828]
test_holiday = np.array(test_holiday)
test_holiday.reshape(7)

for i in range(7):
    if(test_holiday[i] == -50000 or test_holiday[i] == -25000):
        predict[i] = model2.predict(predict[i].reshape(-1,1))
    predict[i] = np.ceil(predict[i])

print('prediction Result')
print(predict.reshape(7))

## save csv
a = [20190402,20190403,20190404,20190405,20190406,20190407,20190408]
dit = {'date':a, 'Peak_Load(MW)':predict.reshape(7)}
df = pd.DataFrame(dit)

df.to_csv(r'./submission.csv',columns=['date','Peak_Load(MW)'],index=False,sep=',')





