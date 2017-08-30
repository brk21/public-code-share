from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf
import time
import unittest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import math
import random
import pandas as pd
import datetime
import io
import os
import sys

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


### PREP DATA FOR TRAINING AND VALIDATION
final_stock_data = pd.read_csv('~/acquisition-prediction/final_stock_data.csv')
final_stock_data['data_date'] = pd.to_datetime(final_stock_data['data_date'])

training_mask = (final_stock_data['data_date'] < datetime.date(year=2014,month=1,day=1)) & (final_stock_data['data_date'] >= datetime.date(year=2003,month=1,day=1))
dev_mask = (final_stock_data['data_date'] >= datetime.date(year=2013,month=1,day=1)) & (final_stock_data['data_date'] < datetime.date(year=2016,month=11,day=1))
test_mask = (final_stock_data['data_date'] >= datetime.date(year=2015,month=1,day=1)) & (final_stock_data['data_date'] < datetime.date(year=2016,month=11,day=1))
#pred_mask = (final_stock_data['data_date'] >= datetime.date(year=2016,month=11,day=1))
#cross_val_mask = (final_stock_data['data_date'] >= datetime.date(year=2004,month=1,day=1)) & (final_stock_data['data_date'] < datetime.date(year=2016,month=1,day=1))

training_set = final_stock_data[training_mask]
dev_set = final_stock_data[dev_mask]
test_set = final_stock_data[test_mask]
#pred_set = final_stock_data[pred_mask]
#cross_val_set = final_stock_data[cross_val_mask]

training_tickers = list(training_set['ticker'].unique()) ### list of tickers
dev_tickers = list(dev_set['ticker'].unique()) ### list of tickers
test_tickers = list(test_set['ticker'].unique()) ### list of tickers
#pred_tickers = list(pred_set['ticker'].unique()) ### list of tickers
#cross_val_tickers = list(cross_val_set['ticker'].unique()) ### list of tickers

x_labs=['close','high','low','volume','days_15_neg_price','days_15_pos_price','days_30_neg_price',
                  'days_30_pos_price','days_60_neg_price','days_60_pos_price','days_90_neg_price',
                  'days_90_pos_price','days_120_neg_price','days_120_pos_price','days_150_neg_price',
                  'days_150_pos_price','days_180_neg_price','days_180_pos_price'] ### which features are we using
n_features = len(x_labs) ### how many features are we using

accuracies = []

def generate_batches(input_df,tickers,tickerlab='ticker',
                             xlabs=x_labs,ylab=['acquisition_pending'],
                             days_per_record=180, new_record_every=10):
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

                    n_days = x_array.shape[0]

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
        
### TO TEST ON THE TEST DATA
#class TestCallback(Callback):
#    def __init__(self, test_data):
#        self.test_data = test_data

#        def on_epoch_end(self, epoch, logs={}):
#            x, y = self.test_data
#            loss, acc = self.model.evaluate(x, y, verbose=1)
#            print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
           
for timesteps_ in [360]:
    for new_record_ in [60]:
        timestepsPerRecord =timesteps_ ### how many days of data per batch. This is the number of timesteps.
        newRecordEvery=new_record_ ### we create a new record every # days. This is the frequency of new record creation.
        ### GENERATE BATCHES FOR EACH SET
        start = time.time()
        x_train, y_train = generate_batches(training_set,training_tickers,days_per_record=timestepsPerRecord, new_record_every=newRecordEvery)
        end = time.time()
        print("Training Data Created")
        print(end - start)

        start = time.time()
        x_dev, y_dev = generate_batches(dev_set,dev_tickers,days_per_record=timestepsPerRecord, new_record_every=newRecordEvery)
        end = time.time()
        print("Dev Data Created")        
        print(end - start)

        start = time.time()
        x_test, y_test = generate_batches(test_set,test_tickers,days_per_record=timestepsPerRecord, new_record_every=newRecordEvery)
        end = time.time()
        print("Test Data Created")      
        print(end - start)

        #start = time.time()
        #x_pred, y_pred = generate_batches(pred_set,pred_tickers)
        #end = time.time()
        #print(end - start)

        #start = time.time()
        #x_cv, y_cv = generate_batches(cross_val_set,cross_val_tickers)
        #end = time.time()
        #print(end - start)

        print("Train Shape")
        print(x_train.shape, y_train.shape)
        print("Percent of Training == 1")        
        print(np.mean(y_train * 1.0))
        print("Dev Shape")
        print(x_dev.shape, y_dev.shape)
        print("Percent of Dev == 1")
        print(np.mean(y_dev * 1.0))
        print("Test Shape")
        print(x_test.shape, y_test.shape)
        print("Percent of Test == 1")                
        print(np.mean(y_test * 1.0))
        
        for batch_size in [150,250]:
            for dropout_rate in [.5]:
                for weight in [100.,125.]:
                    class_weight = {0 : 1.,1: weight} 
                    for num_epochs in [3,4,5,6,7,8]:
                        today = datetime.datetime.now().strftime('%Y_%m_%d_%I%M%p')
                        filename = 'modeling_outputs_' + today + '.txt'
                        # Use join instead of hard-coding file separators
                        path = os.path.join(os.getcwd(), filename)
                        with open(path, 'a+') as file:
                            file.write("\n")
                            file.write(today + " Modeling Outputs \n")
                            file.write(str(timesteps_) + " Timesteps \n")
                            file.write(str(new_record_) + " days between records \n")
                            file.write(str(batch_size) + " batch size \n")
                            file.write(str(dropout_rate) + " droupout rate \n")
                            file.write(str(weight) + ' class weight \n')
                            file.write(str(num_epochs) + ' epochs \n')

                            print(str(timesteps_) + " Timesteps")
                            print(str(new_record_) + " days between records")
                            print(str(batch_size) + " batch size")
                            print(str(dropout_rate) + " droupout rate")
                            print(str(weight) + ' class weight')
                            print(str(num_epochs) + ' epochs')

                            num_samples = x_train.shape[0] - (x_train.shape[0] % batch_size)

                            ### CREATE THE ACTUAL MODEL
                            model = Sequential()
                            # Adding the input layer and the LSTM layer
                            # model.add(LSTM(units = 240, return_sequences = True, input_shape = (timestepsPerRecord, 18)))
                            # Adding a second LSTM layer
                            model.add(LSTM(units = 60, return_sequences = True, input_shape = (timestepsPerRecord, n_features)))
                            model.add(BatchNormalization())
                            # Adding third LSTM layer with bi-directional
                            model.add(LSTM(units = 40, return_sequences = True))
                            model.add(BatchNormalization())
                            model.add(LSTM(units = 30, return_sequences = True))
                            model.add(BatchNormalization())
                            model.add(LSTM(units = 30, return_sequences = True))
                            model.add(BatchNormalization())
                            model.add(LSTM(units = 30))
                            model.add(BatchNormalization())
                            model.add(Dropout(dropout_rate))

                            # we can think of this chunk as the hidden layer    
                            # model.add(Dense(90, kernel_initializer='uniform'))
                            # model.add(BatchNormalization())
                            # model.add(Activation('tanh'))
                            # model.add(Dropout(0.5))

                            # we can think of this chunk as the output layer
                            model.add(Dense(1, kernel_initializer="uniform"))
                            model.add(BatchNormalization())
                            model.add(Activation('sigmoid'))


                            # try using different optimizers and different optimizer configs
                            model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

                            print('Training Model with ', timestepsPerRecord, " Timesteps in each Record and ", newRecordEvery, " Day Gaps Between New Records. Batch Size = ", batch_size, " Dropout Rate = ", dropout_rate , "Class 1 Weight = ", weight )
                            model.fit(x_train[:num_samples], y_train[:num_samples],
                                      batch_size=batch_size,
                                      epochs=num_epochs,
                                      class_weight = class_weight,
                                      validation_data=(x_dev, y_dev))

                            scores = model.evaluate(x_test, y_test, verbose=1)
                            accuracies.append(scores)

                            file.write("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))                    
                            file.write("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

                            print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))                    
                            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

                            file.write("\n")
                            file.write(str(scores))
                            file.write("\n")
                            print(scores)
                            
                            try:
                                model_name = 'model_' + str(timestepsPerRecord) + '_day_' + str(newRecordEvery) + '_gap' + str(num_epochs) +  '_epoch_' + str(dropout_rate) + 'dropout_' + str(weight) + 'weight_5_layer_lstm_' + today + '.h5'
                                model.save(model_name)  # creates a HDF5 file 'my_model.h5'
                                y_preds = model.predict(x_test)
                                for threshold in [.41,.43,.45,.47,.49,.5,.51,.53,.55,.57,.59,.61]:
                                    y_pred = [1 if x > threshold else 0 for x in y_preds]
                                    C = metrics.confusion_matrix(y_test,y_pred)
                                    print("Model: ", model_name)
                                    print("Threshold: ", threshold)
                                    print(C)
                                    file_name = model_name[:-3] + '_' + str(threshold)[1:]
                                    show_confusion_matrix(file_name, C, ['Not Pending Acquisition', 'Pending Acquisition'])
                                del model  # deletes the existing model
                            except:
                                pass


    # returns a compiled model
    # identical to the previous one
    # model = load_model('~/acquisition-prediction/model_name_here.h5')



















