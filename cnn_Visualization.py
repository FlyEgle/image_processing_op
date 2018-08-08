import tensorflow as tf
import codecs
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, Dense
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import json
import matplotlib.pyplot as plt
import keras.backend as K
from PIL import Image
import numpy as np
import h5py
import time
import cv2

train_path = '/home/jiangmingchao/Gan_tensorflow/flower_dataset/train_dataset/'
validation_path = '/home/jiangmingchao/Gan_tensorflow/flower_dataset/validation_dataset/'
model_path = '/home/jiangmingchao/Gan_tensorflow/flower_tf/model/'
if os.path.exists(model_path):
    print('model save path is : %s' % model_path)
else:
    os.mkdir(model_path)
learning_rate_base = 0.01
epochs = 200
model_save_name = 'flower_inception.h5'


def staris_decay(epoch):
    if epoch < 100:
        return learning_rate_base
    elif epoch < 150:
        return 0.001
    elif epoch < 200:
        return 0.0005

def poly_decay(epoch):
    maxEpochs = epochs
    baselr = learning_rate_base
    power = 1.0

    alpha = baselr * (1 - (epoch / float(maxEpochs))) * power
    return alpha


learning_shedule = LearningRateScheduler(staris_decay)
modelckpt = ModelCheckpoint(model_path + model_save_name, 'val_loss', save_best_only=True)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=5,
        horizontal_flip=True
)
validation_datagen = ImageDataGenerator(
        rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
)


def keras_inference(input_shape):
    input_layers = Input(input_shape)
    conv1 = Conv2D(64, (3, 3), activation='relu')(input_layers)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool1 = MaxPooling2D((2, 2), (2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool1)
    conv4 = Conv2D(128, (3, 3), activation='relu')(conv3)
    pool2 = MaxPooling2D((2, 2), (2, 2))(conv4)
    conv5 = Conv2D(256, (3, 3), activation='relu')(pool2)
    conv6 = Conv2D(256, (3, 3), activation='relu')(conv5)
    conv7 = Conv2D(256, (3, 3), activation='relu')(conv6)
    pool3 = MaxPooling2D((2, 2), (2, 2))(conv7)
    conv8 = Conv2D(512, (3, 3), activation='relu')(pool3)
    conv9 = Conv2D(512, (3, 3), activation='relu')(conv8)
    conv10 = Conv2D(512, (3, 3), activation='relu')(conv9)

    flatten = Flatten()(conv10)
    dense1 = Dense(512, activation='relu')(flatten)
    dense2 = Dense(256, activation='relu')(dense1)
    output = Dense(5)(dense2)
    vgg_model = Model(input_layers, output)
    return vgg_model


def inception_v3_model(input_shape):
    model = InceptionV3(include_top=True,
                        weights=None,
                        input_shape=input_shape,
                        classes=5)
    return model


def get_loss_fig(epochs):
    data = []
    with codecs.open("history_logs.json", "r", "utf-8") as f:
        for line in f:
            dic = json.loads(line)
            data.append(dic)
    logs = data[0]
    loss = logs['loss']
    val_loss = logs['val_loss']
    x = [i+1 for i in range(epochs)]
    plt.figure(0)
    plt.plot(x, loss, 'r-', label='loss')
    plt.plot(x, val_loss, 'y-', label='val_loss')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('flower_loss.png')

def proprecess_image(img_path):
    image = Image.open(img_path)
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, 0)
    return image

def get_model(model, output_layers):
    # orginal model inputs as input and layers output as outputs for new model
    show_model = Model(inputs=model.inputs, outputs=output_layers.output)
    return show_model


def get_model_by_layers(model, layer_id, weights_path):
    layers_dict = {}
    for index, layer in enumerate(model.layers):
        layers_dict[index] = layer.name
    
    def generate_model(model, layer_id, weights_path):
        func_layer = model.get_layer(name=layers_dict[layer_id], index=layer_id)
        new_model = get_model(model, func_layer)
        new_model.load_weights(weights_path, by_name=True)
        return new_model

    # get conv layers 
    if layers_dict[layer_id].split('_')[0] == "conv2d":
        conv_model = generate_model(model, layer_id, weights_path)
        return conv_model
    elif layers_dict[layer_id].split('_')[0] == "max_pooling2d":
        pool_model = generate_model(model, layer_id, weights_path)
        return pool_model
    elif layers_dict[layer_id].split('_')[0] == "activation":
        activation_model = generate_model(mdoel, layer_id, weights_path)
        return activation_model


# show conv result not kernel result
def visualize_model_output(model, image_path, layer_id, weights_path, num_filter=8):
    image = proprecess_image(image_path)
    output_model = get_model_by_layers(model, layer_id, weights_path)
    print("======output model summary ======")
    print(output_model.summary())
    result = output_model.predict(image)

    for i in range(num_filter):
        plt.subplot(2, 4, i+1)
        plt.imshow(result[0, :, :, i])
        plt.title(layer_id)
    plt.show()

# conv kernel output
def visulaize_kernel_output(model, image_path, layer_id, weights_path):
    image = proprecess_image(image_path)
    img_shape = image.shape
    layers_dict = {}
    for index, layer in enumerate(model.layers):
        layers_dict[index] = layer.name

    def deprocess_image(x):
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.1

        x += 0.5
        x = np.clip(x, 0, 1)

        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x 

    def normalize(x):
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

    conv_model = get_model_by_layers(model, layer_id, weights_path)
    print(conv_model.summary())

    input_img = conv_model.input
    kept_filters = []
    for filter_index in range(32):
        print('Processing filter %d'%filter_index)
        start_time = time.time()

        layer_output = conv_model.get_layer(name=layers_dict[layer_id]).output
        if K.image_data_format() == 'channel_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])
        
        grads = K.gradients(loss, input_img)[0]

        grads = normalize(grads)

        iterate = K.function([input_img], [loss, grads])

        step = 1

        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((img_shape[0], img_shape[3], img_shape[1], img_shape[2]))
        else:
            input_img_data = np.random.random((img_shape[0], img_shape[1], img_shape[2], img_shape[3]))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('current loss value:', loss_value)
            if loss_value<=0.:
                break
        
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' %(filter_index, end_time - start_time))

    print('filters number: ', len(kept_filters))
    n = 3

    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[: n*n]

    margin = 5

    width = n * image.shape[1] + (n - 1) * margin
    height = n * image.shape[2] + (n - 1) * margin

    stritched_filters = np.zeros((width, height, 3))

    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i *n + j]
            stritched_filters[(image.shape[1] + margin) *i : (image.shape[1] + margin) * i + image.shape[1],
                            (image.shape[2] + margin) * j: (image.shape[2] + margin) * j + image.shape[2], :] = img
    plt.imshow(stritched_filters)
    plt.show()


# visulaize heat map on image
def visualize_heat_map_on_image(model, image_path):
    image = proprecess_image(image_path)
    preds = model.predict(image)
    class_idx = np.argmax(preds[0])
    print(model.output)
    class_output = model.output[:, class_idx]
    # get the conv feature from last convolution
    last_conv_layer = model.get_layer(name='conv2d_188')
    
    # calculate grads
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=[0, 1, 2])
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    # pool grad
    pooled_grads_value, conv_layer_output_value = iterate([image])
    # for i in range()
    for i in range(192):
        conv_layer_output_value[: ,:, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    original_image = cv2.imread(image_path)
    # original_image = cv2.resize(original_image, (224, 224))
    heatmap_image = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_image = np.uint8(255 * heatmap_image)

    heatmap_image = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_HSV)
    print(heatmap_image.shape)
    print(original_image.shape)

    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap_image, 0.4, 0)
    plt.imshow(superimposed_img)
    plt.show()
    

print("=======model summary=======")
with tf.device('/cpu:0'):
    inception_model = inception_v3_model((224, 224, 3))
print(inception_model.summary())

parallel_model = multi_gpu_model(inception_model, gpus=2)

training=False

if training:
    print("=======model.fit===========")
    parallel_model.compile(loss='categorical_crossentropy',
                       optimizer=Adam(0.01),
                       metrics=['accuracy'])
    history_logs = parallel_model.fit_generator(
            train_generator,
            steps_per_epoch=120,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[learning_shedule]
    )
    with codecs.open('history_logs.json', 'w', 'utf-8') as outfile:
        json.dump(history_logs.history, outfile, ensure_ascii=False)
        outfile.write('\n')
    get_loss_fig(epochs)

    print("==========save model =========")
    inception_model.save('flower_inceptionv3.h5')

# predict 
else:
    print("==========load model =========")
    inception_model = InceptionV3(include_top=True, weights=None, input_shape=(224, 224, 3), classes=5)

    # layers name
    for i, layer in enumerate(inception_model.layers):
        print(i, layer.name)

    # print(inception_model.input_layers)
    image_path = '/home/jiangmingchao/Gan_tensorflow/flower_dataset/train_dataset/daisy/105806915_a9c13e2106_n.jpg'
    weights_path = 'flower_inceptionv3.h5'
    conv_layers_id = [155, 187, 232, 290]
    # for layer_id in conv_layers_id:
    #     visualize_model_output(inception_model, image_path, layer_id, weights_path, num_filter=8)
    # for layer_id in conv_layers_id:
    #     visulaize_kernel_output(inception_model, image_path, layer_id, weights_path)
    
    inception_model.load_weights('flower_inceptionv3.h5')
    print(inception_model.output)

    # grad cmp
    visualize_heat_map_on_image(inception_model, image_path)
    
    


