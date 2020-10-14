#========================================================================================
## MLP // B2(with_syslog=False), M2(with_syslog=True)
#========================================================================================
train_balancing=False
test_balancing=False
data_name = {mozilla:'mozilla', chrome:'chrome', firefox:'firefox', eclipse:'eclipse'}
total_split = {'firefox': [[3,7],[3,10,30,90],[3,7,20,50],[1,2,3]], 'chrome':[[3,7],[3,10,30,90],[1,2,5,10],[1,2,3]], 'mozilla':[], 'eclipse':[[3,7],[3,10,30,90],[1,2,3,5],[1,2,3]]} 
time_feature = False
test_mode = False

embedding_concat=False
one_hot_concat = False

with_syslog = False
only_syslog = True

max_activity_len = 10
min_activity_len = 1 # workday 1-10
embedding_size=50
lambda_list = [2]

for lamb in lambda_list:
    class_std = lamb
    for turn in range(2,3):
        if turn==0:
            with_syslog = False
            only_syslog = False
        elif turn==1:
            with_syslog = True
            only_syslog = False
        elif turn==2:
            with_syslog = False
            only_syslog = True
        f1_list = []
        for i in range(6,11):
            totalLength = len(all_data)
            print('Total length: ', totalLength)
            splitLength = int(totalLength / (numCV + 1))
            
            # Split cross validation set
            print ('CV',i)
            print('class_std',class_std)
            if test_mode:
                train_data = all_data[:30]#i*splitLength]
                train_history = all_history[:30]#i*splitLength]
                train_3to1 = sum([len(r) for r in train_data])       
                train_time = all_time[:train_3to1]#i*splitLength]
                train_workday = all_workday[:train_3to1]
                train_recentday = all_recentday[:train_3to1]
                #train_openday = all_openday[:train_3to1]
                train_activitycnt = all_activitycnt[:train_3to1]
            else:
                train_data = all_data[:i*splitLength]
                train_history = all_history[:i*splitLength]
                train_3to1 = sum([len(r) for r in train_data])       
                train_time = all_time[:train_3to1]#i*splitLength]
                train_workday = all_workday[:train_3to1]
                train_recentday = all_recentday[:train_3to1]
                #train_openday = all_openday[:train_3to1]
                train_activitycnt = all_activitycnt[:train_3to1]

            # ===================================================================== 
            updated_train_data = []    
            updated_train_history = []    
            updated_train_time = []

            j=0
            for bug1, bug2 in zip(train_data, train_history):
                train_data_list = []
                train_history_list = []
                if len(bug1)>=min_activity_len:
                    for act1, act2 in zip(bug1, bug2):
                        current_train_filter = [word for word in act1 if word in vocabulary]
                        his_current_train_filter = [word for word in act2 if word in vocabulary]
                        train_data_list.append(current_train_filter)
                        train_history_list.append(his_current_train_filter)
                        updated_train_time.append(train_time[j])
                        j+=1
                    updated_train_data.append(train_data_list)
                    updated_train_history.append(train_history_list)
                else:
                    j+=len(bug1)

            del train_data, train_time, train_history, #train_bugid, train_workday
            gc.collect()

            # ===================================================================== 
            updated_train_time = [0 if x<=class_std else 1 for x in updated_train_time]
            label_num=max(updated_train_time)+1
            updated_train_data_1d = [y for x in updated_train_data for y in x]
            updated_train_history_1d = [y for x in updated_train_history for y in x]
            train_col_size = [len(x) for x in updated_train_data]
            train_n = len(updated_train_data_1d)
            train_df =pd.DataFrame({'x1': updated_train_data_1d, 'x2': updated_train_history_1d, 
                                    'y1': updated_train_time})
            del updated_train_time, updated_train_history, updated_train_data, updated_train_data_1d, updated_train_history_1d
            gc.collect()

            # =============================
            # BALANCE RESAMPLING
            if train_balancing:
                class_count = train_df.y1.value_counts()
                print('\nClass 0:', class_count[0])
                print('Class 1:', class_count[1])
                df_class_0 = train_df[train_df['y1']==0]
                df_class_1 = train_df[train_df['y1']==1]
                if class_count[0]<class_count[1]:
                    under_class_cnt = class_count[0]
                    df_class_1 = df_class_1.sample(under_class_cnt)
                else:
                    under_class_cnt = class_count[1]
                    df_class_0 = df_class_0.sample(under_class_cnt)
                train_df = pd.concat([df_class_0, df_class_1], axis=0)
                train_n = len(train_df)
                print(train_df[:10],'\n')
                print('\nRandom under-sampling:')
                print(train_df.y1.value_counts(),'\n')

            X_train = np.empty(shape=[train_n, max_sentence_len, embed_size_word2vec], dtype='float32') # len(updated_train_data) # train_len*2
            X_train_history = np.empty(shape=[train_n, max_his_len, embed_size_word2vec], dtype='float32') # len(updated_train_history)
            Y_train = np.empty(shape=[train_n,1], dtype='int32') # len(updated_train_time)

            j=0
            for curr_row1, curr_row2 in zip(train_df.x1.values, train_df.x2.values):
                if len(curr_row1)>max_sentence_len:
                    start_loc = len(curr_row1) - max_sentence_len
                else: 
                    start_loc = 0
                sequence_cnt = 0    
                for item1 in curr_row1[start_loc:]:
                    if combined_word2vec:
                        if item1 in vocabulary:
                            X_train[j, sequence_cnt, :] = wordvec_model[item1] 
                            sequence_cnt = sequence_cnt + 1                
                            if sequence_cnt == max_sentence_len-1:
                                break  
                    else:
                        if item1 in vocabulary_1:
                            X_train[j, sequence_cnt, :] = wordvec_model_1[item1] 
                            sequence_cnt = sequence_cnt + 1                
                            if sequence_cnt == max_sentence_len-1:
                                break  
                for k in range(sequence_cnt, max_sentence_len):
                    X_train[j, k, :] = np.zeros((1,embed_size_word2vec))   
                if len(curr_row2)>max_his_len:
                    start_loc = len(curr_row2) - max_his_len
                else: 
                    start_loc = 0
                sequence_cnt = 0
                for item2 in curr_row2[start_loc:]:
                    if combined_word2vec:
                        if item2 in vocabulary:
                            X_train_history[j, sequence_cnt, :] = wordvec_model[item2] 
                            sequence_cnt = sequence_cnt + 1                
                            if sequence_cnt == max_his_len-1:
                                    break 
                    else:
                        if item2 in vocabulary_2:
                            X_train_history[j, sequence_cnt, :] = wordvec_model_2[item2] 
                            sequence_cnt = sequence_cnt + 1                
                            if sequence_cnt == max_his_len-1:
                                    break 
                for k in range(sequence_cnt, max_his_len):
                    X_train_history[j, k, :] = np.zeros((1,embed_size_word2vec))
                #X_train_stream[j,:] = train_df.x3.values[j]
                #X_train_workday[j,:] = train_df.x4.values[j]-1
                Y_train[j,0] = train_df.y1.values[j]
                j+=1
            print(train_df[:30])


            y_train = np_utils.to_categorical(Y_train, label_num)
            del Y_train
            gc.collect()

            X_train_doc = np.zeros(shape=[train_n, embed_size_word2vec], dtype='float32')
            X_train_his_doc = np.zeros(shape=[train_n, embed_size_word2vec], dtype='float32')
            for j,words in enumerate(X_train):
                cnt=0
                smt = np.asarray([0.0 for p in range(embed_size_word2vec)])
                for word in words:
                    if not np.any(word): # All-zero element?
                        break
                    smt+=word
                    cnt+=1
                if not np.any(smt):
                    avg = smt
                else:
                    avg = smt/cnt
                X_train_doc[j] = avg
            for j,words in enumerate(X_train_history):
                cnt=0
                smt = np.asarray([0.0 for p in range(embed_size_word2vec)])
                for word in words:
                    if not np.any(word): # All-zero element?
                        break
                    smt+=word
                    cnt+=1
                if not np.any(smt):
                    avg = smt
                else:
                    avg = smt/cnt
                X_train_his_doc[j] = avg

            if with_syslog:
                X_train_avg = np.concatenate((X_train_doc, X_train_his_doc),axis=1)
            elif only_syslog:
                X_train_avg= X_train_his_doc
            else:
                X_train_avg = X_train_doc
            X_train_concat = X_train_avg

            # =====================================================================
            label_num=max(train_df.y1.values)+1
            del train_df
            gc.collect()

            embedding_input=Input(shape=(len(X_train_concat[0]),), dtype='float32')
            middle_dense = Dense(230, name='middle_dense_1')(embedding_input) 
            middle_dense = LeakyReLU(alpha=0.3)(middle_dense)
            middle_dense = Dense(300, name='middle_dense_2')(middle_dense) 
            middle_dense = LeakyReLU(alpha=0.2)(middle_dense)
            middle_dense = Dense(180, name='middle_dense_3')(middle_dense) 
            middle_dense = LeakyReLU(alpha=0.5)(middle_dense)
            last_dense = Dense(100, name='middle_dense_4')(middle_dense) 
            last_dense = LeakyReLU(alpha=0.1)(last_dense)
            output = Dense(label_num, activation='softmax', name='output')(last_dense)

            model = Model(input=[embedding_input], output=output)
            adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) 
            model.summary()

            # =====================================================================
            # TRAIN MODEL
            es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
            X_train_concat, X_valid_concat, y_train, y_valid = train_test_split(X_train_concat, y_train, test_size=0.15, shuffle= True)
            hist = model.fit(X_train_concat, y_train, batch_size=batch_size, 
                             validation_data = (X_valid_concat, y_valid), epochs=20, callbacks=[es])
            train_result = hist.history
            print('\nTrain_result\n')
            print(train_result)
            del X_train_concat, X_valid_concat, y_train, y_valid
            gc.collect()
            
            # Save model
            model_json = model.to_json()
            if turn==0:
                mode_name = 'b2'
            elif turn==1:
                mode_name = 'M2'
            elif turn==2:
                mode_name = 'only_syslog'
            model_name = data_name[path]+"_"+mode_name+"_dense_noBal_class"+str(class_std)+"_cv"+str(i)+"model.json"
            weight_name = data_name[path]+"_"+mode_name+"_dense_noBal_class"+str(class_std)+"_cv"+str(i)+"model.h5"
            with open(model_name,"w") as json_file :
                json_file.write(model_json)
                model.save_weights(weight_name)
            print("Saved model to disk\n\n\n")    
            
            #========================================================================================
            # TEST DATA
            #========================================================================================
            if test_mode:
                test_data = all_data[30:50]#i*splitLength:(i+1)*splitLength] 
                test_history = all_history[30:50]#i*splitLength:(i+1)*splitLength]
                test_3to1 = sum([len(r) for r in test_data])
                test_time = all_time[train_3to1:train_3to1+test_3to1]#i*splitLength:(i+1)*splitLength]
                #test_stream = all_stream[train_3to1 : train_3to1+test_3to1]
                test_workday = all_workday[train_3to1:train_3to1+test_3to1]
                test_recentday = all_recentday[train_3to1:train_3to1+test_3to1]
                #test_openday = all_openday[train_3to1:train_3to1+test_3to1]
                test_activitycnt = all_activitycnt[train_3to1:train_3to1+test_3to1]
            else:
                test_data = all_data[i*splitLength:(i+1)*splitLength]
                test_history = all_history[i*splitLength:(i+1)*splitLength]
                test_3to1 = sum([len(r) for r in test_data])       
                test_time = all_time[train_3to1 : train_3to1+test_3to1]#i*splitLength]
                #test_stream = all_stream[train_3to1:train_3to1+test_3to1]
                test_workday = all_workday[train_3to1:train_3to1+test_3to1]
                test_recentday = all_recentday[train_3to1:train_3to1+test_3to1]
                #test_openday = all_openday[train_3to1:train_3to1+test_3to1]
                test_activitycnt = all_activitycnt[train_3to1:train_3to1+test_3to1]

            updated_test_data = []
            updated_test_history = []
            updated_test_time = []
            #updated_test_stream = []
            #updated_test_workday = []
            #updated_test_bugid = []

            j=0
            for bug1, bug2 in zip(test_data, test_history):
                test_data_list = []
                test_history_list = []
                if len(bug1)>=min_activity_len:
                    for act1, act2 in zip(bug1, bug2):
                        current_test_filter = [word for word in act1 if word in vocabulary]
                        his_current_test_filter = [word for word in act2 if word in vocabulary]
                        test_data_list.append(current_test_filter)
                        test_history_list.append(his_current_test_filter)
                        updated_test_time.append(test_time[j])
                        #updated_test_stream.append(test_stream[j])
                        #updated_test_workday.append(test_workday[j])
                        #updated_test_bugid.append(test_bugid[j])
                        j+=1
                    updated_test_data.append(test_data_list)
                    updated_test_history.append(test_history_list)
                else:
                    j+=len(bug1)

            del test_data, test_time, test_history,# test_stream, test_bugid
            gc.collect()

            # ===================================================================== 
            updated_test_time = [0 if x<=class_std else 1 for x in updated_test_time]
            curr_split = total_split[data_name[path]][0]
            updated_test_workday = [0 if x<=curr_split[0] else 1 if x<=curr_split[1] else 2 for x in test_workday]
            updated_test_recentday = []
            for x in test_recentday:
                tmp = 0 
                for j,y in enumerate(x):
                    tmp+= pow(2,j)*y
                updated_test_recentday.append(tmp) 
            #curr_split = total_split[data_name[path]][1]
            #updated_test_openday = [0 if x<=curr_split[0] else 1 if x<=curr_split[1] else 2 if x<=curr_split[2] else 3 if x<=curr_split[3] else 4 for x in test_openday]
            curr_split = total_split[data_name[path]][2]
            updated_test_activitycnt = [0 if x<=curr_split[0] else 1 if x<=curr_split[1] else 2 if x<=curr_split[2] else 3 if x<=curr_split[3] else 4 for x in test_activitycnt]
            updated_test_data_1d = [y for x in updated_test_data for y in x]
            updated_test_history_1d = [y for x in updated_test_history for y in x]
            test_col_size = [len(x) for x in updated_test_data]
            test_n = len(updated_test_data_1d)
            del updated_test_data, updated_test_history
            gc.collect()

            # ===================================================================== 
            X_test_last = np.zeros(shape=[test_n, max_sentence_len, embed_size_word2vec], dtype='float32') # len(updated_test_data) # test_len*2
            X_test_history = np.zeros(shape=[test_n, max_his_len, embed_size_word2vec], dtype='float32') # len(updated_test_history)
            y_test = np.empty(shape=[test_n,1], dtype='int32') # len(updated_test_time)

            j=0
            for curr_row1, curr_row2 in zip(updated_test_data_1d, updated_test_history_1d):
                if len(curr_row1)>max_sentence_len:
                    start_loc = len(curr_row1) - max_sentence_len
                else: 
                    start_loc = 0
                sequence_cnt = 0    
                for item1 in curr_row1[start_loc:]:
                    if item1 in vocabulary:
                        X_test_last[j, sequence_cnt, :] = wordvec_model[item1] 
                        sequence_cnt = sequence_cnt + 1                
                        if sequence_cnt == max_sentence_len-1:
                            break  
                for k in range(sequence_cnt, max_sentence_len):
                    X_test_last[j, k, :] = np.zeros((1,embed_size_word2vec))   
                if len(curr_row2)>max_his_len:
                    start_loc = len(curr_row2) - max_his_len
                else: 
                    start_loc = 0
                sequence_cnt = 0
                for item2 in curr_row2[start_loc:]:
                    if item2 in vocabulary:
                        X_test_history[j, sequence_cnt, :] = wordvec_model[item2] 
                        sequence_cnt = sequence_cnt + 1                
                        if sequence_cnt == max_his_len-1:
                                break 
                for k in range(sequence_cnt, max_his_len):
                    X_test_history[j, k, :] = np.zeros((1,embed_size_word2vec))
                '''
                X_test_stream[j,:] = test_df.x3.values[j]
                if test_df.x4.values[j]>9:
                    X_test_workday[j,:] = 9
                else:
                    X_test_workday[j,:] = test_df.x4.values[j]
                '''
                y_test[j,0] = updated_test_time[j]
                j+=1

            X_test_doc = np.zeros(shape=[test_n, embed_size_word2vec], dtype='float32')
            X_test_his_doc = np.zeros(shape=[test_n, embed_size_word2vec], dtype='float32')
            for j,words in enumerate(X_test_last):
                cnt=0
                smt = np.asarray([0.0 for p in range(embed_size_word2vec)])
                for word in words:
                    if not np.any(word): # All-zero element?
                        break
                    smt+=word
                    cnt+=1
                if not np.any(smt):
                    avg = smt
                else:
                    avg = smt/cnt
                X_test_doc[j] = avg
            for j,words in enumerate(X_test_history):
                cnt=0
                smt = np.asarray([0.0 for p in range(embed_size_word2vec)])
                for word in words:
                    if not np.any(word): # All-zero element?
                        break
                    smt+=word
                    cnt+=1
                if not np.any(smt):
                    avg = smt
                else:
                    avg = smt/cnt
                X_test_his_doc[j] = avg

            if with_syslog:
                X_test_avg = np.concatenate((X_test_doc, X_test_his_doc),axis=1)
            elif only_syslog:
                X_test_avg = X_test_his_doc
            else:
                X_test_avg = X_test_doc

            if one_hot_concat:
                X_test_workday = np.asarray(updated_test_workday, dtype=np.float32)
                label_num = max(updated_test_workday)+1
                X_test_workday = np_utils.to_categorical(X_test_workday, label_num)
                X_test_recentday = np.asarray(updated_test_recentday, dtype=np.float32)
                label_num = max(updated_test_recentday)+1
                X_test_recentday = np_utils.to_categorical(X_test_recentday, label_num)
                X_test_activitycnt = np.asarray(updated_test_activitycnt, dtype=np.float32)
                label_num = max(updated_test_activitycnt)+1
                X_test_activitycnt = np_utils.to_categorical(X_test_activitycnt, label_num)

                X_test_concat = np.concatenate((X_test_avg, X_test_workday),axis=1)
                X_test_concat = np.concatenate((X_test_concat, X_test_recentday),axis=1)
                X_test_concat = np.concatenate((X_test_concat, X_test_activitycnt),axis=1)
            elif embedding_concat:
                embedding_test_output = []
                for p in range(4):    
                    model_1 = Model(input=[last_input, history_input], output=middle_dense)
                    loss_ = 'categorical_crossentropy'
                    metrics_ = metrics.categorical_accuracy
                    y_name = 'y'+str(p+1)
                    print('===================='+y_name+'====================')

                    # Load weight
                    weight_name = data_name[path]+"_emb_"+y_name+"_lastActivity_noise"+str(noise_day)+"_classMed"+str(class_std)+"_cv"+str(i)+"model.h5"
                    model_1.load_weights(weight_name, by_name=True)
                    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                    model_1.compile(loss=loss_, optimizer=adam, metrics=[metrics_])    
                    embedding_test_output.append(model_1.predict([X_test_last, X_test_history]))
                    del model_1

                X_test_concat = np.concatenate((X_test_avg, embedding_test_output[0]),axis=1)
                X_test_concat = np.concatenate((X_test_concat, embedding_test_output[1]),axis=1)
                X_test_concat = np.concatenate((X_test_concat, embedding_test_output[2]),axis=1)
                X_test_concat = np.concatenate((X_test_concat, embedding_test_output[3]),axis=1)
            else:
                X_test_concat = X_test_avg
            #========================================================================================
            # PREDICT & ACCURACY
            #========================================================================================
            predict = model.predict([X_test_concat]) 
            predictY = np.argmax(predict, axis=1)
            corrects = np.nonzero(predictY.reshape((-1,1)) == y_test) #.reshape((-1,))
            accu = len(corrects[0])/len(X_test_concat)
            print("Total test accuracy : %.2f" % (accu*100))
            print('\nConfusion Matrix')
            cm = confusion_matrix(y_test, predictY)
            print(cm)
            prec = cm[0,0]/(cm[0,0]+cm[1,0])
            recall = cm[0,0]/(cm[0,0]+cm[0,1])
            f1 = 2*prec*recall/(prec+recall)
            f1_list.append(f1)
            print('f1 score: %.2f' %(f1*100))
            print('precision: %.2f' %(prec*100 ))
            print('recall: %.2f' %(recall*100)) 
            total = sum(cm[0])+sum(cm[1])
            print('Predict 0: %.2f' % ((cm[0,0]+cm[1,0])/total*100))
            print('Predict 1: %.2f' % ((cm[0,1]+cm[1,1])/total*100))
            print()

            print('\n%.2f %.2f %.2f %.2f\n' %(prec*100, recall*100, f1*100, accu*100))
            del model
            gc.collect()
