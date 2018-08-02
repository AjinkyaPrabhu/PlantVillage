from Model import create_model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler, TensorBoard, EarlyStopping




##model = create_model(len(data1[1][1]))
##compile and fit model on dataset

##model.compile()




WIDTH,HEIGHT = 256,256
train_data_dir = './crowdai/'
N_CLASSES = 38
nb_train_samples = 4125
batch_size = 16
epochs = 10



model = create_model(N_CLASSES)

model.compile(loss = "categorical_crossentropy",
              optimizer = optimizers.Adam(lr=0.01),
              metrics =["accuracy"] )

print(model.summary())
#
# train_datagen = ImageDataGenerator(horizontal_flip = True,
#                                    fill_mode = "nearest",
#                                    zoom_range = 0.3,
#                                    width_shift_range = 0.3,
#                                    height_shift_range=0.3,
#                                    rotation_range=30)

train_datagen = ImageDataGenerator()



train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                   target_size = (WIDTH, HEIGHT),
                                                   batch_size = batch_size,
                                                   class_mode = "categorical")
checkpoint = ModelCheckpoint("leaf_checkpoint.h5",
                             monitor='val_acc',
                             verbose=1, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', period=1)


board = TensorBoard(log_dir="./logs_mobilenet/",
                    write_graph=True,
                    batch_size=batch_size,
                    write_images=True,)



model.fit_generator(train_generator,
                    epochs = epochs,
                    callbacks = [board,checkpoint])


model.save("Whale_model_1.h5")
