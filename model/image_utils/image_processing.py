# -*- coding:utf-8 -*-

# @Time    : 19-2-28 下午10:53

# @Author  : Swing


import tensorflow as tf


def distort_image(image, thread_id):
    with tf.name_scope('flip_horizontal', values=[image]):
        image = tf.image.random_flip_left_right(image)

    # # Randomly distort the colors based on thread id.
    color_ordering = thread_id % 2
    with tf.name_scope("distort_color", values=[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.032)

        image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def process_image(encoded_image,
                  is_training,
                  height,
                  width,
                  resize_height=346,
                  resize_width=346,
                  thread_id=0,
                  image_format='jpeg'):
    def image_summary(name, image):
        if not thread_id:
            tf.summary.image(name, tf.expand_dims(image, 0))

    # Image decoding。shape=[?, ?, 3] range [0, 1]
    with tf.name_scope('decode', values=[encoded_image]):
        if image_format == 'jpeg':
            image = tf.image.decode_jpeg(encoded_image, channels=3)
        elif image_format == 'png':
            image = tf.image.decode_png(encoded_image, channels=3)
        else:
            raise ValueError('Invalid image format：　%s' % image_format)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image_summary('original_image', image)

    # Resize image.
    assert (resize_height > 0) == (resize_width > 0)
    if resize_height:
        image = tf.image.resize_images(image,
                                       size=[resize_height, resize_width],
                                       method=tf.image.ResizeMethod.BILINEAR)

        # Crop to final dimensions.
    if is_training:
        image = tf.random_crop(image, [height, width, 3])
    else:
        # Central crop, assuming resize_height > height, resize_width > width.
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)

    image_summary('resized_image', image)

    if is_training:
        image = distort_image(image, thread_id)

    image_summary('final_image', image)

    # Rescale to [-1,1] instead of [0, 1]
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image
