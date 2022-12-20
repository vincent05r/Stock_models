import tensorflow as tf


class aux_functions():

    def scheduler_1(epoch, lr):
        if epoch < 8:
            return lr
        elif epoch < 40:
            return lr * tf.math.exp(-0.05)
        else:
            return lr



    #inspect time series window
    def window_inspection(t_window, mode, label=['open']):
        if mode == True:
            
            for i in label:
                t_window.plot(plot_col = i)
            
            print(t_window.train.element_spec)
            
            for example_inputs, example_labels in t_window.train.take(1):
                print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
                print(f'Labels shape (batch, time, features): {example_labels.shape}')

    

    #compile and fit using tensorflow data window
    def compile_and_fit(model, window, epochs=20, es_patience=4, es_monitor='val_loss', es_mode = 'min', lr = 0.001):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = es_monitor,
                                                        patience = es_patience,
                                                        mode = es_mode)

        optimizer_adam = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=optimizer_adam,
                    metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

        multistep_lr = tf.keras.callbacks.LearningRateScheduler(aux_functions.scheduler_1)

        history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping, multistep_lr])
        
        print("Learning rate of the model: ", optimizer_adam._decayed_lr('float32').numpy())

        model.summary()
        
        return history