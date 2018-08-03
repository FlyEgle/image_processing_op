import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from calculate_convolution.padding_op import padding_op
from functools import reduce

img = cv2.imread("1.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# image = np.ones((5, 5))
image = np.array([[1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0]])
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

kernel_size = kernel.shape[0]
strides = 2
not_padding = True
shape = image.shape


# calculate patch * kernel instead of the real convolution operate
def patch_calculate_sum(patch, kernel):
    return np.sum(np.multiply(patch, kernel))


# get patch of image
def get_patch_array(image, center_points, kernel_size):
    center_size = int(kernel_size / 2)
    x, y = center_points
    min_points = [x - center_size, y - center_size]
    max_points = [x + center_size, y + center_size]
    patch = image[min_points[0]: max_points[0] + 1, min_points[1]: max_points[1] + 1]
    return patch


# new image size with nopadding
def generate_new_size_output(image_shape, kernel_size, strides):
    center_slide_width = int(np.ceil((image_shape[0] - kernel_size + 1) / strides))
    center_slide_height = int(np.ceil((image_shape[1] - kernel_size + 1) / strides))

    output_array = np.zeros((center_slide_width, center_slide_height))
    return output_array


def not_padding_no_strides(image, kernel, strides):
    output_array = generate_new_size_output(image.shape, kernel.shape[0], strides)
    for i in range(output_array.shape[0]):
        for j in range(output_array.shape[1]):
            center_points = (i + int(kernel.shape[0] / 2) + int(strides / 2),
                             j + (int(kernel.shape[0] / 2) + int(strides / 2)))
            patch = get_patch_array(image, center_points, kernel.shape[0])
            convolution_value = patch_calculate_sum(patch, kernel)
            if convolution_value >= 255:
                convolution_value = 255
            elif convolution_value <= 0:
                convolution_value = 0
            output_array[i][j] = convolution_value
    return output_array


# TODO: WRONG
def no_padding_strides(image, kernel, strides):
    output_array = generate_new_size_output(image.shape, kernel.shape[0], strides)
    for i in range(shape[0]):
        if i % strides == 0 and i < shape[0] - 1:
            for j in range(shape[0]):
                if j % strides == 0 and j < shape[0] - 1:
                    center_points = (i + int(kernel.shape[0] / 2), j + (int(kernel.shape[0] / 2)))
                    patch = get_patch_array(image, center_points, kernel.shape[0])
                    convolution_value = patch_calculate_sum(patch, kernel)
                    if convolution_value >= 255:
                        convolution_value = 255
                    elif convolution_value <= 0:
                        convolution_value = 0
                    a = int(i / strides)
                    b = int(j / strides)
                    output_array[a][b] = convolution_value
    return output_array


# TODO: this is wrong
def padding_no_strides(image, kernel_size, strides, is_padding):
    image = padding_op(image, is_padding, "same", [1, 1, 1, 1])
    padding_array = not_padding_no_strides(image, kernel_size, strides)
    return padding_array


def padding_strides(image, kernel, is_padding, padding, strides):
    image = padding_op(image, is_padding, "same", padding)
    padding_strides_array = not_padding_no_strides(image, kernel, strides)
    return padding_strides_array


def generate_conv_weights(kernel_size, input_channels, output_channels):
    kernel_weights = np.random.randn(kernel_size, kernel_size, input_channels, output_channels)
    return kernel_weights


def conv3d_for_inputchannel(image, weights_shape, kernel_weights, is_padding, padding, strides):
    input_channels = weights_shape[2]
    # input_channels is same as image.shape[2]
    output_image_filter = []
    for i in range(input_channels):
        kernel_2d = kernel_weights[:, :, i]
        padding_strides_filter = padding_strides(image[:, :, i], kernel_2d, is_padding, padding, strides=strides)
        output_image_filter.append(padding_strides_filter)
    return output_image_filter


def conv2d(image, weights_shape, is_padding, padding, strides):
    kernel_size = weights_shape[0]
    input_channels = weights_shape[2]
    output_channels = weights_shape[3]

    kernel_weights = generate_conv_weights(kernel_size, input_channels, output_channels)

    assert image.shape[2] == input_channels
    image_filter = []
    for i in range(output_channels):
        image_feature = conv3d_for_inputchannel(image, weights_shape, kernel_weights[:, :, :, i],
                                                is_padding, padding, strides)
        sum_image_feature = reduce(lambda x, y: x+y, image_feature)
        print(type(sum_image_feature), sum_image_feature.shape)
        shape = sum_image_feature.shape
        sum_image_feature = np.expand_dims(sum_image_feature, 2).reshape(shape[0], shape[1], 1)
        image_filter.append(sum_image_feature)
    output_featuremap = np.concatenate(image_filter, axis=2)
    return output_featuremap


# image = cv2.imread("1.jpg")
# image = cv2.resize(image, (360, 360))
#
# out_feature = conv2d(image, [3, 3, 3, 10], False, padding=[1, 1, 1, 1], strides=1)
#
# for i in range(4):
#     plt.figure(i)
#     plt.imshow(out_feature[:, :, i])
#     plt.show()


output_array = not_padding_no_strides(image, kernel, 1)
# output_array2 = no_padding_strides(img_gray, kernel_size, 2)
# output_image_array = not_padding_no_strides(img_gray, kernel_size, 1)
# output_image_array_strides = no_padding_strides(img_gray, kernel_size, 2)
# output_padding_no_strides = padding_no_strides(img_gray, kernel_size, 1)
# output_padding_strides = padding_strides(img_gray, kernel_size, 2)
#
print(output_array)
# print(output_array2)
#
# print(output_image_array.shape)
# plt.subplot(221)
# plt.imshow(output_image_array)
# plt.title("no padding no strides, shape is {}".format(output_image_array.shape))
#
# img_gray = np.expand_dims(img_gray, 2)
# output_image_array2 = cv2.filter2D(img_gray, -1, kernel)
# plt.subplot(222)
# plt.imshow(output_image_array2)
# plt.title("cv2. filter2D, shape is {}".format(output_image_array2.shape))
#
# print(output_padding_no_strides.shape)
# print(np.max(output_padding_no_strides))
# plt.subplot(223)
# plt.imshow(output_padding_no_strides)
# plt.title("padding no strides, shape is {}".format(output_padding_no_strides.shape))
#
# print(output_padding_strides.shape)
# print(np.max(output_padding_strides))
# plt.subplot(224)
# plt.imshow(output_array2)
# plt.title("no padding and strides, shape is {}".format(output_array2.shape))
# plt.show()


