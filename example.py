from signate_sony import data_make, model
import pandas as pd
import numpy as np
import pickle

"""feature_y, feature_t, label, df_tr = data_make.to_data("train.csv", pd.read_csv("train.csv"), Top_k=2)
model_t = model.model_one(label, feature_t, "model_t.pkl", verbose_eval=1, early_stopping_rounds=1)
model_y = model.model_one(label, feature_y, "model_y.pkl", verbose_eval=1, early_stopping_rounds=1)
model_ty = model.model_both(label, feature_y, feature_t, "model_ty.pkl", verbose_eval=1, early_stopping_rounds=1)
"""
feature_y, feature_t, label, df_tr = data_make.to_data("train.csv", pd.read_csv("test.csv"), Top_k=2)
t=np.all(np.concatenate([np.expand_dims(model.None_ex(feature_y),1),np.expand_dims(model.None_ex(feature_t),1)],axis=1),axis=1)
feature=np.concatenate([feature_y[t],feature_t[t,47:]],axis=1)
preds_ty = pickle.load(open('model_ty.pkl', 'rb')).predict(feature)
preds_y = pickle.load(open('model_y.pkl', 'rb')).predict(feature_y[model.None_ex(feature_y)])
preds_t = pickle.load(open('model_t.pkl', 'rb')).predict(feature_t[model.None_ex(feature_t)])

y=0
t=0
ty=0
list_=np.array([])
T=model.None_ex(feature_t).astype(np.int)-model.None_ex(feature_y).astype(np.int)
for i in T:
    if i == -1:
        list_=np.append(list_,preds_y[y])
        y+=1
    elif i == 1:
        list_=np.append(list_,preds_t[t])
        t+=1
    elif i == 0:
        list_=np.append(list_,preds_ty[ty])
        ty+=1
        
df=pd.read_csv("submit_sample.csv", header=None)
df[1]=list_
df.to_csv("submit_sample1.csv",header=False, index=False)