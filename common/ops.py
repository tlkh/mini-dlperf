import tensorflow as tf

def crop_center_and_resize(image, img_size):
    s = tf.shape(image)
    w, h = s[0], s[1]
    c = tf.minimum(w, h)
    w_start = (w - c) // 2
    h_start = (h - c) // 2
    center = tf.slice(image, [w_start, h_start, 0], [c, c, -1])
    return tf.image.resize(image, (img_size, img_size))

def resize_preserve_ratio(image, img_size):
    s = tf.cast(tf.shape(image), tf.float32)
    c = tf.minimum(s[0], s[1])
    new_s = tf.cast(s * img_size / c, tf.int32)
    return tf.image.resize(image, (new_s[0], new_s[1]))

def augment_image(image, img_size):
    image = tf.image.random_crop(image, (img_size, img_size, 3))
    image = tf.image.random_brightness(image, max_delta=32/255)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_flip_left_right(image)
    return image