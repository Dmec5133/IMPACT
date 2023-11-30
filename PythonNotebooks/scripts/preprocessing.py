from sklearn.model_selection import train_test_split
import pandas as pd
import random
def data_prep(raw_csv):
    t_csv = raw_csv.transpose()
    raw_csv = t_csv.fillna('Target')
    #print(raw_csv)
    df_list = {k:v.transpose() for k,v in raw_csv.groupby('Phylum')}
    dict_ = {}
    for k,v in df_list.items():
        dict_[k] = pd.merge(v.drop('Phylum'), df_list['Target'],left_index=True, right_index=True).head(-1)
        dict_.pop('Target',None)
    return dict_
def strat_train_test(in_dict, t_size):
    train_dict = {}
    test_dict = {}
    for i ,(k,v) in enumerate(in_dict.items()):
        if i == 0:
            trainX = v.drop(['y'], axis=1).astype(float)
            train_col = v['y']

            X_train, X_test, y_train, y_test = train_test_split(trainX, train_col,test_size=t_size, random_state=random.randint(1,250), stratify = train_col)
            test_index = X_test.index
            train_index = X_train.index
            train_dict[k]= v.loc[train_index,:]
            test_dict[k]= v.loc[test_index,:]
        else:
            train_dict[k]= v.loc[train_index,:]
            test_dict[k]= v.loc[test_index,:]
    return train_dict, test_dict
