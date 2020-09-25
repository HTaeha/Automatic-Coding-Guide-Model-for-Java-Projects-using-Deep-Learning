import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def return_the_number_of_label_data(label_data):
    zero = 0
    one = 0
    for item in label_data:
        if item == 0:
            zero += 1
        else:
            one += 1
    return zero, one

def return_the_number_of_predict_data(predict):
    choose_zero = 0
    choose_one = 0
    for data in predict:
        if data == 1:
            choose_one += 1
        else:
            choose_zero += 1
    return choose_zero, choose_one
    
def accuracy(predict, Y):
    true = np.sum(Y == predict)
    accuracy = (float(true)/len(predict))*100
    return accuracy

def f1_score(test_label_data, predict):
    f1_score = metrics.f1_score(test_label_data, predict)
    return f1_score

def AUC(test_label_data, preds):
    fpr, tpr, threshold = roc_curve(test_label_data, preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def print_confusion_matrix(Y_test, predict):
    print("\nConfusion Matrix")
    cm = confusion_matrix(Y_test, predict)
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for p in range(cm.shape[0]):
        print('True label', p)
        for q in range(cm.shape[0]):
            print(cm[p,q], end=' ')
            if q%100 == 0:
                print(' ')
        print(' ')

def save_predict_rate(TP_file_name, FP_file_name, TN_file_name, FN_file_name, predict, predict_rate, test_label_data, test_path, test_method):
 #TP, FP, TN, FN file save.
    f_TP = open(TP_file_name, 'w')
    f_FP = open(FP_file_name, 'w')
    f_TN = open(TN_file_name, 'w')
    f_FN = open(FN_file_name, 'w')

    for i, data in enumerate(test_label_data):
        if data == 1:
            if predict[i] == 1:
                f_TP.write(test_path[i] + '\r\n')
                f_TP.write(test_method[i]+ '\r\n')
                f_TP.write(str(predict_rate[i,0]) + '\t' + str(predict_rate[i,1]) + '\r\n')
            else:
                f_FN.write(test_path[i]+ '\r\n')
                f_FN.write(test_method[i]+ '\r\n')
                f_FN.write(str(predict_rate[i,0]) + '\t' + str(predict_rate[i,1]) + '\r\n')
        else :
            if predict[i] == 1:
                f_FP.write(test_path[i]+ '\r\n')
                f_FP.write(test_method[i]+ '\r\n')
                f_FP.write(str(predict_rate[i,0]) + '\t' + str(predict_rate[i,1]) + '\r\n')
            else:
                f_TN.write(test_path[i]+ '\r\n')
                f_TN.write(test_method[i]+ '\r\n')
                f_TN.write(str(predict_rate[i,0]) + '\t' + str(predict_rate[i,1]) + '\r\n')

    f_TP.close()
    f_FP.close()
    f_TN.close()
    f_FN.close()

def save_ROC(roc_count, test_label_data, preds, roc_name):
    fpr, tpr, threshold = roc_curve(test_label_data, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(roc_count)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(roc_name)

def histogram(input_file, data, vocab, max_sentence_len, x_range):
    hist = [0 for i in range(max_sentence_len//x_range + 1)]
#    hist = []
    for j, curr_row in enumerate(data):
        sentence_len_count = 0
        for item in curr_row:
            if item in vocab:
                sentence_len_count += 1
                if sentence_len_count == max_sentence_len - 1:
                    break
#hist.append(sentence_len_count)
        if sentence_len_count == max_sentence_len - 1:
            hist[-1] += 1
        else:
            hist[sentence_len_count//x_range] += 1
    list_x = []
    for i in range(len(hist)):
        list_x.append(i*x_range)
    plt.bar(list_x, hist, width = 1)
    for i,j in zip(list_x,hist):
        plt.text(i, j, str(j), fontsize = 8)
#plt.hist(hist, histtype = 'bar', rwidth = 0.9, bins = 20)
    plt.title('histogram of '+input_file)
    plt.xlabel('sentence length')
    plt.ylabel('frequency')
    path = 'histogram/histogram-'+input_file
    plt.savefig(path)

def histogram_ground_truth(input_file, data, label, max_sentence_len, x_range, num):
    hist_zero = [0 for i in range(max_sentence_len//x_range + 1)]
    hist_one = [0 for i in range(max_sentence_len//x_range + 1)]
    for j, curr_row in enumerate(data):
        sentence_len_count = len(curr_row)
        if sentence_len_count >= max_sentence_len:
            if label[j] == 0:
                hist_zero[-1] += 1
            else:
                hist_one[-1] += 1
        else:
            if label[j] == 0:
                hist_zero[sentence_len_count//x_range] += 1
            else:
                hist_one[sentence_len_count//x_range] += 1
                
    list_x = []
    for i in range(len(hist_zero)):
        list_x.append(i*x_range)
    color = ['black', 'green', 'red']
    plt.figure(num, figsize=(30,15))
    plt.bar(list_x, hist_one, width = 2)
    plt.bar(list_x, hist_zero, width = 2, bottom = hist_one)
    idx = 0
    '''
    for i,j,k in zip(list_x,hist_zero,hist_one):
        if j > 3000:
            plt.text(i, j, str(j+k), color = color[idx], fontsize = 15)
            plt.text(i, j/2, str(j), color = color[idx], fontsize = 15)
        else:
            plt.text(i, 1500, str(j+k), color = color[idx],fontsize = 15)
            plt.text(i, 750, str(j), color = color[idx], fontsize = 15)
        plt.text(i, 10, str(k), color = color[idx], fontsize = 15)
        if idx < 2:
            idx += 1
        else:
            idx = 0
    '''
    for i,j,k in zip(list_x,hist_zero,hist_one):
        plt.text(i, 100, str(j+k), color = color[idx], fontsize = 15)
        plt.text(i, 50, str(j), color = color[idx], fontsize = 15)
        plt.text(i, 0, str(k), color = color[idx], fontsize = 15)
        if idx < 2:
            idx += 1
        else:
            idx = 0
    hist_name = "histogram-"+input_file+"_ground_truth_num"
    plt.title(hist_name)
    plt.xlabel('sentence length')
    plt.ylabel('frequency')
    path = 'histogram/last/'+hist_name
    plt.savefig(path)

    hist_zero_prob = [0 for i in range(max_sentence_len//x_range + 1)]
    hist_one_prob = [0 for i in range(max_sentence_len//x_range + 1)]
    for i, item in enumerate(hist_zero):
        total = item + hist_one[i]
        if total == 0:
            hist_zero_prob[i] = 0
            hist_one_prob[i] = 0
            continue
        hist_zero_prob[i] = (item/total) * 100
        hist_one_prob[i] = (hist_one[i]/total) * 100
    plt.figure(num+1, figsize=(30,15))
    plt.bar(list_x, hist_one_prob, width=2)
    plt.bar(list_x, hist_zero_prob, width=2, bottom = hist_one_prob)
    for i,j in zip(list_x, hist_zero_prob):
        plt.text(i, 50, str(int(j)), fontsize = 15)
    for i,j in zip(list_x, hist_one_prob):
        plt.text(i, 0, str(int(j)), fontsize = 15)
    hist_name = "histogram-"+input_file+"_prediction_rate"
    plt.title(hist_name)
    plt.xlabel('sentence length')
    plt.ylabel('frequency')
    path = 'histogram/last/'+hist_name
    plt.savefig(path)

def histogram_predict(input_file, data, x_range, num):
    plt.figure(num)
    hist = [0 for i in range(x_range)]
    for j, curr_row in enumerate(data):
        hist[int(data[j][0]*10)] += 1

    list_x = []
    for i in range(len(hist)):
        list_x.append(i*x_range)

    plt.bar(list_x, hist, width = 1)
    for i,j in zip(list_x,hist):
        plt.text(i, j, str(j), fontsize = 8)
    plt.title('histogram of '+input_file)
    plt.xlabel('sentence length')
    plt.ylabel('frequency')
    path = 'histogram/predict/histogram-0_'+input_file
    plt.savefig(path)

    plt.figure(num+1)
    hist = [0 for i in range(x_range)]
    for j, curr_row in enumerate(data):
        hist[int(data[j][1]*10)] += 1

    list_x = []
    for i in range(len(hist)):
        list_x.append(i*x_range)

    plt.bar(list_x, hist, width = 1)
    for i,j in zip(list_x,hist):
        plt.text(i, j, str(j), fontsize = 8)
    plt.title('histogram of '+input_file)
    plt.xlabel('sentence length')
    plt.ylabel('frequency')
    path = 'histogram/predict/histogram-1_'+input_file
    plt.savefig(path)

