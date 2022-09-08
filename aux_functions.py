import tensorflow as tf


class aux_functions():

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
    def compile_and_fit(model, window, epochs=20, es_patience=4, es_monitor='val_loss', es_mode = 'min'):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = es_monitor,
                                                        patience = es_patience,
                                                        mode = es_mode)

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

        history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
        
        model.summary()
        
        return history