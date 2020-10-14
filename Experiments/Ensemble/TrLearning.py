import pandas as pd
import random
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn import tree
from sklearn import linear_model

import numpy as np

def tradaboost(trans_S, trans_A, label_S, label_A, test, N):
    trans_data = np.concatenate((trans_A, trans_S), axis=0)
    trans_label = np.concatenate((label_A, label_S), axis=0)


    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    test_data = np.concatenate((trans_data, test), axis=0)

    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0)

    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])

    predict = np.zeros([row_T])

    print('params initial finished.')

    trans_data = np.asarray(trans_data, order='C')
    trans_label = np.asarray(trans_label, order='C')
    test_data = np.asarray(test_data, order='C')

    print(trans_data.shape, trans_label.shape)

    for i in range(N):

        P = calculate_P(weights, trans_label)

        result_label[:, i] = train_classify(trans_data, trans_label, test_data, P)

        print('result:', result_label[:, i], row_A, row_S, i, result_label.shape)

        error_rate = calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],weights[row_A:row_A + row_S, :])
        print('Error rate:', error_rate)
        if error_rate > 0.5:
            error_rate = 0.5
        if error_rate == 0:
            N = i
            break

        bata_T[0, i] = error_rate / (1 - error_rate)
        print(bata_T)
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],(-np.abs(result_label[row_A + j, i] - label_S[j])))

        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - label_A[j]))

    for i in range(row_T):

        left = np.sum(result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

        if left >= right:
            predict[i] = 1
        else:
            predict[i] = 0

    return predict


def calculate_P(weights, label):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


def train_classify(trans_data, trans_label, test_data, P):
    clf = linear_model.SGDClassifier()
    # print(trans_label)
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    pred = clf.predict(test_data)
    # for i in range(len(pred)):
    #     if(abs(pred[i]-101)>=abs(pred[i]-112)):
    #         pred[i] = 1
    #     else:
    #         pred[i] = 0
    return pred


def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)

    # print(weight[:] / total)
    # print(np.abs(label_R - label_H))
    return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))


def append_feature(dataframe, istest):
    lack_num = np.asarray(dataframe.isnull().sum(axis=1))
    if istest:
        X = dataframe.values
        X = X[:, 1:X.shape[1]]
    else:
        X = dataframe.values
        X = X[:, 1:X.shape[1]]
    total_S = np.sum(X, axis=1)
    var_S = np.var(X, axis=1)
    X = np.c_[X, total_S]
    X = np.c_[X, var_S]
    X = np.c_[X, lack_num]

    return X


train_df = pd.DataFrame(pd.read_csv("mushroom_e.csv"))
train_df.fillna(value=-999999)
train_df1 = pd.DataFrame(pd.read_csv("mushroom_t.csv"))
# print(len(train_df_ex))
# train_df1 = train_df_ex.sample(frac=0.1, replace=False, random_state=1)
# print(len(train_df1))
train_df1.fillna(value=-999999)
# test_df = pd.DataFrame(pd.read_csv("mushroom_t.csv"))
# test_df.fillna(value=-999999)

print(train_df.shape, train_df1.shape)
print(train_df, train_df1)

le = preprocessing.LabelEncoder()

for col in train_df.columns:
    train_df[col] = le.fit_transform(train_df[col])

for col in train_df1.columns:
    train_df1[col] = le.fit_transform(train_df1[col])

train_data_T = train_df.values
train_data_S = train_df1.values
# test_data_S = test_df.values


# for i in range(len(test_data_S)):
#     for c in range(len(test_data_S[i])):
#         test_data_S[i][c] = ord(test_data_S[i][c])

print('data loaded.')

# label_T = train_data_T[:, train_data_T.shape[1] - 1]
# trans_T = append_feature(train_df, istest=False)
#
# label_S = train_data_S[:, train_data_S.shape[1] - 1]
# trans_S = append_feature(train_df1, istest=False)
#
# test_data_no = test_data_S[:, 0]
# test_data_S = append_feature(test_df, istest=True)

label_T = train_data_T[:, 0]
trans_T = train_data_T[:, 1:]
# trans_T = append_feature(train_df, istest=False)

label_S = train_data_S[:, 0]
trans_S = train_data_S[:, 1:]
# trans_S = append_feature(train_df1, istest=False)

# test_data_no = train_data_S[:, 0]
# test_data_no = test_data_no.astype('int')
# test_data_S = append_feature(test_df, istest=True)

print('data split end.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape)

scaler = preprocessing.StandardScaler()

trans_T =scaler.fit_transform(trans_T)
trans_S =scaler.fit_transform(trans_S)

# imputer_T = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# imputer_S = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
#
# imputer_S.fit(trans_S, label_S)
#
# trans_T = imputer_S.transform(trans_T)
# trans_S = imputer_S.transform(trans_S)

# test_data_S = imputer_S.transform(test_data_S)


print('data preprocessed.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape)

X_train, X_test, y_train, y_test = model_selection.train_test_split(trans_S, label_S, test_size=0.90, random_state= 42)
print('data form.', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(len(X_test))
pred = tradaboost(X_train, trans_T, y_train, label_T, X_test, 30)
print(pred.shape, pred)
true = 0
for i, data in enumerate(pred):
    if data == y_test[i]:
        true += 1
print('accuracy:', true/len(pred))
fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=pred, pos_label=1)
print('auc:', metrics.auc(fpr, tpr))
