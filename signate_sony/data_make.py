import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
tqdm.pandas()

def to_data(path, df_te, Top_k=4):
    """データ作成

    Args:
        path (str): データが保存されてるpath
        df_te (DataFrame): 入力するデータ
        Top_k (int, optional): 近傍の都市の個数. Defaults to 4.

    Returns:
        _type_: _description_
    """
    global df_tr
    df_tr = pd.read_csv(path)
    train_city = df_tr["City"].unique()
    test_city = df_te["City"].unique()
    global train_city_lonlat
    train_city_lonlat = {k: make_lon_lat(df_tr, k) for k in train_city}
    global test_city_lonlat
    test_city_lonlat = {k: make_lon_lat(df_te, k) for k in test_city}
    df_tr["date"] = df_tr.progress_apply(lambda df: datetime.datetime(df["year"],df["month"],df["day"]), axis=1)
    df_te["date"] = df_te.progress_apply(lambda df: datetime.datetime(df["year"],df["month"],df["day"]), axis=1)
    df_tr = df_tr.drop(['year', 'month', 'day'], axis=1)
    df_te = df_te.drop(['year', 'month', 'day'], axis=1)
    global id_to_city
    id_to_city = df_tr.set_index("id")[["City"]].to_dict()["City"]
    df_te["near_city_y"] = df_te.progress_apply(search_city_y, X=(df_tr, Top_k), axis=1)
    df_te["near_city_t"] = df_te.progress_apply(search_city_t, X=(df_tr, Top_k), axis=1)
    feature_y, feature_t, label = make_dataset(df_te, Top_k)
    return feature_y, feature_t, label, df_te

def make_lon_lat(df, city):
    """対象の都市の緯度経度を出力

    Args:
        df (DataFrame): 入力データ
        city (str): 対象の都市

    Returns:
        list: [緯度,経度]
    """
    lat = df.loc[df["City"]==city, "lat"].values[0]
    lon = df.loc[df["City"]==city, "lon"].values[0]
    return [lat,lon]

def make_distance(city, cityid_list):
    """緯度軽度から距離を計算

    Args:
        city (str): 対象の都市
        cityid_list (list): 都市のidリスト

    Returns:
        np.array: 全都市の距離
    """
    city_lonlat = {id_to_city[k]: train_city_lonlat[id_to_city[k]] for k in cityid_list}
    city_lonlat[city]=test_city_lonlat[city]
    radian_all = np.radians(np.array(list(city_lonlat.values())))
    sin_all = np.sin(radian_all) @ np.array([[0,0],[1,0]])
    cos_all = np.cos(radian_all) @ np.array([[0,0],[0,1]])
    x_1 = sin_all + cos_all
    radian_city = np.radians(np.array(city_lonlat[city]))
    radian_del = (radian_all - radian_city) @ np.array([[0,1],[0,0]])
    sin_city = np.sin(radian_city) @ np.array([[0,0],[1,0]])
    cos_city = np.cos(radian_city) @ np.array([[0,0],[0,1]]) * np.cos(radian_del)
    x_2 = sin_city + cos_city
    
    a = x_1@x_2.T
    return np.arccos(np.diag(np.where(a > 1 ,1, a)))

def make_near_city(city, cityid_list, Top_k):
    """付近の都市を探索

    Args:
        city (str): 対象の都市
        cityid_list (list): 全都市のid
        Top_k (int): 探索する個数

    Returns:
        np.array: 付近の都市のidのリスト
    """
    if len(cityid_list)==0:
        return None
    distance_list = make_distance(city, cityid_list)
    if len(distance_list)>=Top_k+1:
        candi_city = np.argsort(distance_list)[1:Top_k+1]
    else:
        candi_city = np.argsort(distance_list)
        candi_city = np.full(Top_k+1-len(distance_list),candi_city[0]).append(candi_city)
    return cityid_list[candi_city]

def search_city_y(df, X):
    """対象のデータの前日の付近の都市を検索

    Args:
        df (DataFrame): 
        X (tuple): (DataFrame, int)

    Returns:
        list: 近傍都市のidリスト
    """
    df_all, Top_k = X
    yesterday = df["date"]-datetime.timedelta(days=1)
    cityid_list = df_all[df_all["date"]==yesterday]["id"].values
    return make_near_city(city=df["City"],cityid_list=cityid_list, Top_k=Top_k)

def search_city_t(df, X):
    """上の翌日ver

    Args:
        df (DataFrame): 
        X (tuple): (DataFrame, int)

    Returns:
        list: 近傍都市のidリスト
    """
    df_all, Top_k = X
    tomorrow = df["date"]+datetime.timedelta(days=1)
    cityid_list = df_all[df_all["date"]==tomorrow]["id"].values
    return make_near_city(city=df["City"],cityid_list=cityid_list, Top_k=Top_k)

def id_to_data(id_, df, Top_k):
    """idに対してデータを出力

    Args:
        id_ (int): データのid
        df (DataFrame): データ
        Top_k (int): 探索する個数

    Returns:
        no.array: 対象のデータ
    """
    if id_ is None:
        return np.zeros(48*Top_k)
    else:
        return df[df["id"].isin(id_)].loc[:, "lat":"pm25_mid"].values.flatten()
    
def make_dataset(df, Top_k):
    """特徴量生成

    Args:
        df (DataFrame): データ
        Top_k (int): 探索する個数

    Returns:
        np.array: 特徴量
    """
    ids_y = df["near_city_y"].values
    ids_t = df["near_city_t"].values
    feature = df.loc[:, "lat":"dew_var"].values
    feature_y = np.array([id_to_data(k, df_tr, Top_k) for k in tqdm(ids_y)])
    feature_t = np.array([id_to_data(k, df_tr, Top_k) for k in tqdm(ids_t)])
    try:
        df['pm25_mid']
    except KeyError as e:
        label = None
    else:
        label = df["pm25_mid"].values
    return np.concatenate([feature, feature_y], 1), np.concatenate([feature, feature_t], 1), label