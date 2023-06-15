## ----------------------------------------------------------------------------
from tensorflow.keras import models, layers, Model
from tensorflow.keras import metrics 
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras import Input

METRICS = ['mae']

##########################################################################################################

def build_model(lr = .0004, conv_filters = 16, 
                dense_neurons = 16, dense_layers = 1, 
                cnn_layers=1, conv_relu_pool_layers = 2, 
                activity_reg = 0.001, input_channels = 2, 
                loss_str='mean_absolute_error', opt=optimizers.Adam(learning_rate=0.1), 
                act_func='relu',
                nlats=18, nlons=45):

    initializer = initializers.HeUniform()
    
    stacked_input = layers.Input(shape=(nlats, nlons, input_channels), name="stacked_input")
    calday_input = layers.Input(shape=(1,), name="calday")
    
    x = stacked_input
    
    for m in range(conv_relu_pool_layers):
        for n in range(cnn_layers):
            x = layers.Conv2D(conv_filters, (3,3), 
                              activity_regularizer=regularizers.l2(activity_reg),
                              kernel_initializer=initializer)(x)
            x = layers.Activation(act_func)(x)
        x = layers.MaxPooling2D((2,2))(x)
                       
    x = layers.Flatten()(x)
    x = layers.concatenate([x, calday_input])
    
    for i in range(dense_layers):
        x = layers.Dense(dense_neurons, 
                         activity_regularizer=regularizers.l2(activity_reg),
                        kernel_initializer=initializer)(x)
        x = layers.Activation(act_func)(x)
         
    tmax_pred = layers.Dense(1, activation='linear')(x) 
    
    model = Model(inputs = [stacked_input, calday_input],
                  outputs = [tmax_pred])
    
    model.compile(loss=loss_str, 
                      optimizer=opt, 
                      metrics=METRICS,
                      weighted_metrics=[])
    
    return(model)


