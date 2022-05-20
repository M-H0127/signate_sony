import numpy as np
from sklearn.model_selection import KFold
import optuna.integration.lightgbm as lgb
import lightgbm as lgbm
import pickle

def None_ex(feature):#ダミーデータを検出
    """ダミーデータを検出

    Args:
        feature (np.array): 各都市の特徴量

    Returns:
        np.array: ダミーでない要素のid
    """
    feature = feature[:,47:]
    return ~np.all(np.where(feature==0, True, False),axis=1)

def model_one(label, feature, save_path, verbose_eval=100, early_stopping_rounds=100, split=5):
    """前日または翌日の付近の都市からの予測

    Args:
        label (np.array): ラベル
        feature (np.array): _description_
        verbose_eval (int, optional): _description_. Defaults to 100.
        early_stopping_rounds (int, optional): _description_. Defaults to 100.
        folds (int, optional): kfoldのsplit数. Defaults to 5.

    Returns:
        model: 完成したモデル
    """
    label=label[None_ex(feature)]
    feature=feature[None_ex(feature)]

    trainval = lgb.Dataset(feature, label)
    params = {'objective': 'regression',
            'metric': 'rmse',
            'random_seed':0} 
    tuner = lgb.LightGBMTunerCV(params, trainval, verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds, folds=KFold(n_splits=split))
    tuner.run()
    params.update(tuner.best_params)
    lgb_ = lgbm.train(params, trainval)
    file = save_path
    pickle.dump(lgb_, open(file, 'wb'))
    return lgb_
        
def model_both(label, feature_y, feature_t, save_path, verbose_eval=100, early_stopping_rounds=100, split=5):
    """#前日と翌日の付近の都市から予測する関数

    Args:
        label (np.array): _description_
        feature_y (np.array): _description_
        feature_t (np.array): _description_
        verbose_eval (int, optional): _description_. Defaults to 100.
        early_stopping_rounds (int, optional): _description_. Defaults to 100.
        folds (int, optional): kfoldのsplit数. Defaults to 5.

    Returns:
        model: 完成したモデル
    """
    t=np.all(np.concatenate([np.expand_dims(None_ex(feature_y),1),np.expand_dims(None_ex(feature_t),1)],axis=1),axis=1)
    feature=np.concatenate([feature_y[t],feature_t[t,47:]],axis=1)
    label=label[t]
    trainval = lgb.Dataset(feature, label)
    params = {'objective': 'regression',
            'metric': 'rmse',
            'random_seed':0} 
    tuner = lgb.LightGBMTunerCV(params, trainval, verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds, folds=KFold(n_splits=split))
    tuner.run()
    params.update(tuner.best_params)
    lgb_ = lgbm.train(params, trainval)
    file = save_path
    pickle.dump(lgb_, open(file, 'wb'))
    return lgb_