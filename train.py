from __future__ import print_function
import  numpy  as  np
import matplotlib.pyplot as plt
from  sklearn.model_selection  import train_test_split
from os import walk, getcwd
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import cv2 as cv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, AveragePooling2D
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import confusion_matrix

batch_size = 128

num_classes = 10

epochs = 40

img_rows, img_cols = 28, 28

mypath = "data/"
txt_name_list = []

slice_train = 30500


def readData():
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    xtotal = []
    ytotal = []
    x_val = []
    y_val = []

    for (dirpath, dirnames, filenames) in walk(mypath):
        if filenames != '.DS_Store':
            txt_name_list.extend(filenames)
            break

    #print(mypath)
    i=0
    for txt_name in txt_name_list:
        txt_path = mypath + txt_name
        x = np.load(txt_path)
        print(txt_name)
        print(i)
        x = x.astype('float32') / 255.  ##scale images
        y = [i] * len(x)
        x = x[:slice_train]
        y = y[:slice_train]

        if i != 0:
            xtotal = np.concatenate((x, xtotal), axis=0)
            ytotal = np.concatenate((y, ytotal), axis=0)
        else:
            xtotal = x
            ytotal = y
        i += 1

    print("xshape = ", xtotal.shape)
    print("yshape = ", ytotal.shape)
    x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.3, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    return x_train, x_val, x_test, y_train, y_val, y_test

def cnnOld(x_train, x_val, x_test, y_train, y_val, y_test):

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # more reshaping
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')
    x_train /= 255
    x_test /= 255
    x_val /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    print(y_train.shape)

    print(input_shape)

    # convert class vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    #
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    ##Normalize the activations of the previous layer at each batch,
    # i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    # Rectified Linear Unit.
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))

    model.add(Activation('softmax'))

    filepath = "saved/weights02.{epoch:02d}.h5"
    ES = EarlyStopping(patience=5)
    check = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    # Adam combines the good properties of Adadelta and RMSprop and hence tend to do better for most of the problems.
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1, validation_data=(x_val, y_val), callbacks=[ES, check])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('cnnVgg.h5')
    print("Saved model to disk")
    #
    # cm = metrics.confusion_matrix(test_batch.classes, y_pred)
    # # or
    # # cm = np.array([[1401,    0],[1112, 0]])
    #
    # plt.imshow(cm, cmap=plt.cm.Blues)
    # plt.xlabel("Predicted labels")
    # plt.ylabel("True labels")
    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.title('Confusion matrix ')
    # plt.colorbar()
    # plt.show()
    print(y_test)

    loaded_model = keras.models.load_model('cnnOld2.h5')
    print("test")
    #y_pred = loaded_model.predict_on_batch(x_test)
    #score = loaded_model.evaluate(x_test, y_test, verbose=0)

    y_pred = loaded_model.predict(x_test)
    print(y_pred)
    # for y in y_pred:
    #     print ("y before change: ", y)
    #     for index in y:
    #         print (" index before change ",index)
    #         if index > 0.5:
    #             index = 1
    #         else:
    #             index = 0
    #         print (" index after change ",index)
    #     print("after change::", y)

    #y_pred = (y_pred > 0.5)
    print("debug")
    indexes = np.argmax(y_pred, axis=1)
    i=0
    for y in y_pred:
        print("original",y)
        y[y<1000]=0
        # print("allzero",y)
        y[indexes[i]] = 1
        print("after",y)
        i+=1
        print("----")


#    y_pred = np.array(np.argmax(y_pred, axis=1))


    #cm = confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(
        y_test.argmax(axis=1), y_pred.argmax(axis=1))
    acc = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), normalize=True, sample_weight=None)
    cr = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    print(acc)
    print(cr)
    #print(y_pred)
    #score = loaded_model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    #model.summary()
    pass


def lenet(x_train, x_val, x_test, y_train, y_val, y_test):
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # more reshaping
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')
    x_train /= 255
    x_test /= 255
    x_val /= 255

    # convert class vectors
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_val = np.pad(x_val, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    print(y_train.shape)

    print(input_shape)

    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=10, activation='softmax'))

    filepath = "saved/weightslenet.{epoch:02d}.h5"
    ES = EarlyStopping(patience=5)
    check = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1, validation_data=(x_val, y_val), callbacks=[ES, check])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('cnnOld2.h5')
    print("Saved model to disk")
    #
    # cm = metrics.confusion_matrix(test_batch.classes, y_pred)
    # # or
    # # cm = np.array([[1401,    0],[1112, 0]])
    #
    # plt.imshow(cm, cmap=plt.cm.Blues)
    # plt.xlabel("Predicted labels")
    # plt.ylabel("True labels")
    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.title('Confusion matrix ')
    # plt.colorbar()
    # plt.show()
    print(y_test)

    loaded_model = keras.models.load_model('cnnlenet.h5')
    print("test")
    #y_pred = loaded_model.predict_on_batch(x_test)
    #score = loaded_model.evaluate(x_test, y_test, verbose=0)

    y_pred = loaded_model.predict(x_test)
    print(y_pred)

    indexes = np.argmax(y_pred, axis=1)
    i=0
    for y in y_pred:
        y[y<1000]=0
        # print("allzero",y)
        y[indexes[i]] = 1
        i+=1

    cm = confusion_matrix(
        y_test.argmax(axis=1), y_pred.argmax(axis=1))
    acc = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), normalize=True, sample_weight=None)
    cr = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    print(acc)
    print(cr)

def main():
    x_train, x_val, x_test, y_train, y_val, y_test = readData()
    lenet(x_train, x_val, x_test, y_train, y_val, y_test)

if __name__ == '__main__':
    main()