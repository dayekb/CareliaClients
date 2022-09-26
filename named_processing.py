import pandas as pd
import urllib.parse

import numpy as np

df_named = pd.read_csv("named.csv")


df_sample_solution = pd.read_csv("sample_solution.csv")

df_train = pd.read_csv("train_dataset_train.csv")

def get_one_hot_encoding(dataset, features):
    temp = [pd.get_dummies(dataset[feature], prefix = feature) for feature in features]
    ohe_df = pd.concat(temp, axis = 1)
    return ohe_df

counts = df_named['url'].value_counts()


# Разбиваем ссылки (по точке)
df_named['split'] = [x.split('.') for x in df_named['url']]
# Считаем длину ссылки (в символах)
df_named['len_url'] = [len(x) for x in df_named['url']]
# Берем предпослений "домен"
df_named['site'] = [x[-2] for x in df_named['split']]

# Берем предпослений "домен" а также, которые до него
df_named['subdom'] = [x[-3] if len(x)>=3 else '' for x in df_named['split'] ]
df_named['subdom2'] = [x[-4] if len(x)>=4 else '' for x in df_named['split'] ]
# Считаем количество "доменов"
df_named['len_split'] = [len(x) for x in df_named['split']]

# Переводим домены в категории (чтобы уменьшить память)
cols = ['site', 'subdom', 'subdom2']
for col in cols:
    df_named[col] = df_named[col].astype('category')
    d = dict(enumerate(df_named[col].cat.categories))
    print(d)
    df_named[col] = df_named[col].cat.codes

# генерируем one-hot представление предпоследних доменов. считаем сумму по contact_id 
ohe = get_one_hot_encoding(df_named, ['site']).join(df_named["contract_id"]).groupby("contract_id").sum()
ohe.to_csv('site_ohe_count.csv')

# Для каждого contact_id считаем статистику длины url в символах (медиана, ско, мин и макс)
loglen_dist = df_named[['len_url',"contract_id"]].groupby(by = "contract_id",).agg( [('median',np.median),
                                                                                     ('std',np.std),
                                                                                     ('min',np.min) ,
                                                                                     ('max',np.max)] )
loglen_dist.to_csv('loglen.csv')

# Для каждого contact_id считаем статистику длины url в различных доменов (медиана, ско, мин и макс)

log_dist = df_named[['len_split',"contract_id"]].groupby(by = "contract_id",).agg( [('median',np.median),
                                                                                     ('std',np.std),
                                                                                     ('min',np.min) ,
                                                                                     ('max',np.max)] )
log_dist.to_csv('log.csv')

# Для каждого contact_id считаем общую статистику посещения сайтов (сколько есть записей)

count_log = df_named[['url',"contract_id"]].groupby("contract_id").count()

count_log.to_csv('url_count.csv')

df_named.drop(['date','url','split','subdom2'],axis = 1).to_csv('url_ohe_count.csv', index = None)
