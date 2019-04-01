# Peak Load Forecasting

Jupyter Notebook https://nbviewer.jupyter.org/gist/tingyin3721/607d713b9c78e5ee064875d5edecdfc5

## 1. Overview
根據台電歷史資料，預測未來七天的"電力尖峰負載"(MW)。

## 2. Goal
預測 2019/4/2 ~ 2019/4/8 的每日"電力尖峰負載"(MW)


## 3. 使用資料
 - [台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)
 - [今日預估尖峰備轉容量率](https://www.taipower.com.tw/d006/loadGraph/loadGraph/load_reserve_.html)
 - [中華民國一百零六年政府行政機關辦公日曆表](https://www.dgpa.gov.tw/information?uid=2&pid=4293)
 - [中華民國一百零七年政府行政機關辦公日曆表](https://www.dgpa.gov.tw/information?uid=83&pid=7473)
 - [中華民國108年（西元2019年）政府行政機關辦公日曆表](https://www.dgpa.gov.tw/information?uid=83&pid=8150)
 
 
## 4. 預測方法
使用 Ensemble Linear Regression進行訓練後預測
 - **1st Regression**
 - Input : 前60 天的每日尖峰負載
 - Predict : 接下來 7 天的每日尖峰負載
 - Training Data : 755
 - Method : sklearn linear regression
 - **2nd Regression**
 - Input : 前兩年所有連假的尖峰負載預測結果
 - Predict : 利用第一次預測結果和真正數值去學習連假的fine-tune regression
 - Training Data : 126
 - Method : sklearn linear regression
 

## 5. 檔案說明
 - data.csv: 所有CSV資料
 - regression.py: 訓練過程.py
 - forecasting_model.m: 1st Regression 訓練模型
 - forecasting_model2.m: 2nd Regression 訓練模型
 - app.py: 讀取訓練模型並進行 2019/4/2 ~ 2019/4/8 預測，並將結果存到 submission.csv
 
 
 