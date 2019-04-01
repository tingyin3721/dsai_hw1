import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.externals import joblib

data_read = pd.read_csv("./data.csv")

print(data_read['尖峰負載(MW)'][0:10])
print(data_read['尖峰負載(MW)'][10:17])

for i in range(755):
    temp = data_read['尖峰負載(MW)'][i+0:i+60]
    # data_used = data_read['淨尖峰供電能力(MW)'][0:365]
    templabel = data_read['尖峰負載(MW)'][i+60:i+67]
    x = np.linspace(0, 1, len(temp))
    x = x.reshape(1, -1)

    # print(data_used)
    # print(label)
    temp = np.array(temp)
    templabel = np.array(templabel)
    temp = temp.reshape(1, -1)
    templabel = templabel.reshape(1, -1)

    if(i==0):
        data_used = temp
        label = templabel
    else:
        #print(data_used.shape)
        #print(temp.shape)
        data_used = np.concatenate((data_used, temp), axis=0)
        label = np.concatenate((label, templabel), axis=0)

model = LinearRegression()
model.fit(data_used, label)

# save models
joblib.dump(model, "forecasting_model.m")

#model = joblib.load("forecasting_model.m")

# regression2
# prepare data
train_2nd = []
label_2nd = []
for i in range(755):
    temp = data_read['尖峰負載(MW)'][i+0:i+60]
    tempholiday = data_read['假日'][i+60:i+67]
    templabel_2nd = data_read['尖峰負載(MW)'][i+60:i+67]

    # print(data_used)
    # print(label)
    temp = np.array(temp)
    tempholiday = np.array(tempholiday)
    temp = temp.reshape(1, -1)
    tempholiday = tempholiday.reshape(7)

    templabel_2nd = np.array(templabel_2nd)
    templabel_2nd = templabel_2nd.reshape(7)

    predict_1st = model.predict(temp)
    predict_1st = predict_1st.reshape(7)

    for j in range(7):
        if(tempholiday[j] == -50000):
            train_2nd.append(predict_1st[j])
            label_2nd.append(templabel_2nd[j])

train_2nd = np.array(train_2nd).reshape(-1,1)
label_2nd = np.array(label_2nd).reshape(-1,1)

print("regression2")
print(train_2nd.shape)
print(label_2nd.shape)

print("start 2nd regression training")
model2 = LinearRegression()
model2.fit(train_2nd, label_2nd)

# save models
joblib.dump(model2, "forecasting_model2.m")

# validation
test = data_read['尖峰負載(MW)'][761:821]
test = np.array(test)
test = test.reshape(1,-1)
predict = model.predict(test)
predict = np.array(predict)
predict = predict.reshape(7)


test_holiday = data_read['假日'][821:828]
test_holiday = np.array(test_holiday)
test_holiday.reshape(7)
#print(test_holiday.shape)
#print(test_holiday[0])
#print("1st")
#print(predict)

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

#df.to_csv(r'./submission.csv',columns=['date','Peak_Load(MW)'],index=False,sep=',')





