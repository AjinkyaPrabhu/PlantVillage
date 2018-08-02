from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,Activation,BatchNormalization



def create_model(n_out_classes = 32, dropout_enable = False):

    # model = mobilenet.MobileNet(input_shape = (256,256,3),
    #                             classes= n_out_classes,
    #                             weights=None
    #                             )
    model = Sequential()
    ##Conv2d Layer
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 3),padding="valid"))
    model.add(Activation('relu'))
    ##3x3 pooling
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu',padding="valid"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    
    if dropout_enable:
        model.add(Dropout(0.20))

    model.add(Flatten())

    model.add(Dense(n_out_classes, activation='softmax'))

    print(model.summary())

    return model


    

    
