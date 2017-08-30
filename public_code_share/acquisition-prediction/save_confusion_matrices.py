import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from keras.models import load_model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import pandas as pd
import datetime
import math
import random
import io

x_labs=['close','high','low','volume','days_15_neg_price','days_15_pos_price','days_30_neg_price',
          'days_30_pos_price','days_60_neg_price','days_60_pos_price','days_90_neg_price',
          'days_90_pos_price','days_120_neg_price','days_120_pos_price','days_150_neg_price',
          'days_150_pos_price','days_180_neg_price','days_180_pos_price']

final_stock_data = pd.read_csv('~/acquisition-prediction/final_stock_data.csv')
final_stock_data['data_date'] = pd.to_datetime(final_stock_data['data_date'])
test_mask = (final_stock_data['data_date'] >= datetime.date(year=2015,month=1,day=1)) & (final_stock_data['data_date'] < datetime.date(year=2016,month=11,day=1))
#pred_mask = (final_stock_data['data_date'] >= datetime.date(year=2016,month=11,day=1))

test_set = final_stock_data[test_mask]
#pred_set = final_stock_data[pred_mask]

test_tickers = list(test_set['ticker'].unique()) ### list of tickers
#pred_tickers = list(pred_set['ticker'].unique()) ### list of tickers

def generate_batches(input_df,tickers,tickerlab='ticker',
                     xlabs=x_labs,ylab=['acquisition_pending'],
                     days_per_record=180, new_record_every=30):
    x_scaled = []
    y_labels = []
    for ticker in tickers:
        mask = (input_df[tickerlab] == ticker)
        ticker_df = input_df[mask]
        
        if ticker_df.empty:
            pass
        else:
            x_df = ticker_df[xlabs]
            y_df = ticker_df[ylab]


            x_array = x_df.as_matrix()

            y_array = y_df.as_matrix()

            n_days = x_array.shape[0] - 1

            last_record = (n_days - days_per_record) + new_record_every
            remainder = last_record % new_record_every

            scaler = MinMaxScaler(feature_range=(0, 1))
            for i in range(remainder,last_record,new_record_every):
                x_record = x_array[i:i+days_per_record,:]
                x_record = scaler.fit_transform(x_record)
                x_scaled.append(x_record)

                y_record = y_array[i:i+days_per_record,:]
                if np.sum(y_record) > 0:
                    y_labels.append(1)
                else:
                    y_labels.append(0)


    y_labels = np.array(y_labels)        

    x_scaled = np.array(x_scaled)

    return (x_scaled,y_labels)


def show_confusion_matrix(filename,C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
#     plt.show()
    fig.savefig(filename + '.png') 
    
models_180_30 = ['model_180_day_30_gap3_epoch_0_5dropout_150_0weight_5_layer_lstm_2017_07_19_0608AM.h5',
                  'model_180_day_30_gap1_epoch_0_5dropout_150_0weight_5_layer_lstm_2017_07_19_0532AM.h5',
                'model_180_day_30_gap2_epoch_0_5dropout_150_0weight_5_layer_lstm_2017_07_19_0545AM.h5',
                'model_180_day_30_gap2_epoch_0_5dropout_100_0weight_5_layer_lstm_2017_07_19_0326AM.h5']

models_180_60 = ['model_180_day_60_gap1_epoch_0_5dropout_150_0weight_5_layer_lstm_2017_07_19_0833AM.h5',
                'model_180_day_60_gap2_epoch_0_5dropout_150_0weight_5_layer_lstm_2017_07_19_0839AM.h5',
                'model_180_day_60_gap3_epoch_0_5dropout_125_0weight_5_layer_lstm_2017_07_19_0815AM.h5']

models_180_90 = ['model_180_day_90_gap2_epoch_0_5dropout_100_0weight_5_layer_lstm_2017_07_19_0951AM.h5',
                'model_180_day_90_gap3_epoch_0_5dropout_100_0weight_5_layer_lstm_2017_07_19_0959AM.h5']

models_240_30 = ['model_240_day_30_gap2_epoch_0_5dropout_100_0weight_5_layer_lstm_2017_07_19_1155AM.h5',
                'model_240_day_90_gap3_epoch_0_5dropout_150_0weight_5_layer_lstm_2017_07_19_0853PM.h5',
                 'model_240_day_30_gap3_epoch_0_5dropout_125_0weight_5_layer_lstm_2017_07_19_0151PM.h5',
                'model_240_day_30_gap2_epoch_0_5dropout_125_0weight_5_layer_lstm_2017_07_19_0122PM.h5',
                'model_240_day_30_gap3_epoch_0_5dropout_100_0weight_5_layer_lstm_2017_07_19_1224PM.h5']

models_240_60 = ['model_240_day_60_gap3_epoch_0_5dropout_150_0weight_5_layer_lstm_2017_07_19_0633PM.h5']

models_300_90 = ['model_300_day_90_gap2_epoch_0_5dropout_150_0weight_5_layer_lstm_2017_07_20_0805AM.h5']

models_360_30 = ['model_360_day_30_gap2_epoch_0_5dropout_100_0weight_5_layer_lstm_2017_07_20_0933AM.h5',
                 'model_360_day_30_gap3_epoch_0_5dropout_100_0weight_5_layer_lstm_2017_07_20_1012AM.h5',
                'model_360_day_30_gap3_epoch_0_5dropout_125_0weight_5_layer_lstm_2017_07_20_1206PM.h5']

models_360_60 = ['model_360_day_60_gap1_epoch_0_5dropout_125_0weight_5_layer_lstm_2017_07_20_0434PM.h5',
                 'model_360_day_60_gap3_epoch_0_5dropout_100_0weight_5_layer_lstm_2017_07_20_0405PM.h5']

x_test, y_test = generate_batches(test_set,test_tickers,days_per_record=180,new_record_every=30)

# for model_name in models_180_30:
#     model = load_model(model_name)
#     y_preds = model.predict(x_test)
#     for threshold in [.41,.43,.45,.47,.49,.5,.51,.53,.55,.57,.59,.61]:
#         y_pred = [1 if x > threshold else 0 for x in y_preds]
#         C = metrics.confusion_matrix(y_test,y_pred)
#         print("Model: ", model_name)
#         print("Threshold: ", threshold)
#         print(C)
#         file_name = model_name[:-3] + '_threshold_' + str(threshold)[1:]
#         show_confusion_matrix(file_name, C, ['Not Pending Acquisition', 'Pending Acquisition'])
        
# x_test, y_test = generate_batches(test_set,test_tickers,days_per_record=180,new_record_every=60)
        
# for model_name in models_180_60:
#     model = load_model(model_name)
#     y_preds = model.predict(x_test)
#     for threshold in [.41,.43,.45,.47,.49,.5,.51,.53,.55,.57,.59,.61]:
#         y_pred = [1 if x > threshold else 0 for x in y_preds]
#         C = metrics.confusion_matrix(y_test,y_pred)
#         print("Model: ", model_name)
#         print("Threshold: ", threshold)
#         print(C)
#         file_name = model_name[:-3] + '_' + str(threshold)[1:]
#         show_confusion_matrix(file_name, C, ['Not Pending Acquisition', 'Pending Acquisition'])
        
# x_test, y_test = generate_batches(test_set,test_tickers,days_per_record=180,new_record_every=90)
        
# for model_name in models_180_90:
#     model = load_model(model_name)
#     y_preds = model.predict(x_test)
#     for threshold in [.41,.43,.45,.47,.49,.5,.51,.53,.55,.57,.59,.61]:
#         y_pred = [1 if x > threshold else 0 for x in y_preds]
#         C = metrics.confusion_matrix(y_test,y_pred)
#         print("Model: ", model_name)
#         print("Threshold: ", threshold)
#         print(C)
#         file_name = model_name[:-3] + '_' + str(threshold)[1:]
#         show_confusion_matrix(file_name, C, ['Not Pending Acquisition', 'Pending Acquisition'])        
        
        
# x_test, y_test = generate_batches(test_set,test_tickers,days_per_record=240,new_record_every=30)
        
# for model_name in models_240_30:
#     model = load_model(model_name)
#     y_preds = model.predict(x_test)
#     for threshold in [.41,.43,.45,.47,.49,.5,.51,.53,.55,.57,.59,.61]:
#         y_pred = [1 if x > threshold else 0 for x in y_preds]
#         C = metrics.confusion_matrix(y_test,y_pred)
#         print("Model: ", model_name)
#         print("Threshold: ", threshold)
#         print(C)
#         file_name = model_name[:-3] + '_' + str(threshold)[1:]
#         show_confusion_matrix(file_name, C, ['Not Pending Acquisition', 'Pending Acquisition'])    
        
# x_test, y_test = generate_batches(test_set,test_tickers,days_per_record=240,new_record_every=60)
        
# for model_name in models_240_60:
#     model = load_model(model_name)
#     y_preds = model.predict(x_test)
#     for threshold in [.41,.43,.45,.47,.49,.5,.51,.53,.55,.57,.59,.61]:
#         y_pred = [1 if x > threshold else 0 for x in y_preds]
#         C = metrics.confusion_matrix(y_test,y_pred)
#         print("Model: ", model_name)
#         print("Threshold: ", threshold)
#         print(C)
#         file_name = model_name[:-3] + '_' + str(threshold)[1:]
#         show_confusion_matrix(file_name, C, ['Not Pending Acquisition', 'Pending Acquisition'])        
        
x_test, y_test = generate_batches(test_set,test_tickers,days_per_record=300,new_record_every=90)
        
for model_name in models_300_90:
    model = load_model(model_name)
    y_preds = model.predict(x_test)
    for threshold in [.41,.43,.45,.47,.49,.5,.51,.53,.55,.57,.59,.61]:
        y_pred = [1 if x > threshold else 0 for x in y_preds]
        C = metrics.confusion_matrix(y_test,y_pred)
        print("Model: ", model_name)
        print("Threshold: ", threshold)
        print(C)
        file_name = model_name[:-3] + '_' + str(threshold)[1:]
        show_confusion_matrix(file_name, C, ['Not Pending Acquisition', 'Pending Acquisition'])         
        
x_test, y_test = generate_batches(test_set,test_tickers,days_per_record=360,new_record_every=30)
        
for model_name in models_360_30:
    model = load_model(model_name)
    y_preds = model.predict(x_test)
    for threshold in [.41,.43,.45,.47,.49,.5,.51,.53,.55,.57,.59,.61]:
        y_pred = [1 if x > threshold else 0 for x in y_preds]
        C = metrics.confusion_matrix(y_test,y_pred)
        print("Model: ", model_name)
        print("Threshold: ", threshold)
        print(C)
        file_name = model_name[:-3] + '_' + str(threshold)[1:]
        show_confusion_matrix(file_name, C, ['Not Pending Acquisition', 'Pending Acquisition'])        
        
x_test, y_test = generate_batches(test_set,test_tickers,days_per_record=360,new_record_every=60)
        
for model_name in models_360_60:
    model = load_model(model_name)
    y_preds = model.predict(x_test)
    for threshold in [.41,.43,.45,.47,.49,.5,.51,.53,.55,.57,.59,.61]:
        y_pred = [1 if x > threshold else 0 for x in y_preds]
        C = metrics.confusion_matrix(y_test,y_pred)
        print("Model: ", model_name)
        print("Threshold: ", threshold)
        print(C)
        file_name = model_name[:-3] + '_' + str(threshold)[1:]
        show_confusion_matrix(file_name, C, ['Not Pending Acquisition', 'Pending Acquisition'])                