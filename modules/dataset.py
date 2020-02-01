import tensorflow as tf
from modules.anchor import encode_tf


def _parse_tfrecord(img_dim, using_bin, using_flip, using_distort,
                    using_encoding, priors, match_thresh, ignore_thresh,
                    variances):
    def parse_tfrecord(tfrecord):
        features = {
            'image/img_name': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark0/x': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark0/y': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark1/x': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark1/y': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark2/x': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark2/y': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark3/x': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark3/y': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark4/x': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark4/y': tf.io.VarLenFeature(tf.float32),
            'image/object/landmark/valid': tf.io.VarLenFeature(tf.float32)}
        if using_bin:
            features['image/encoded'] = tf.io.FixedLenFeature([], tf.string)
            x = tf.io.parse_single_example(tfrecord, features)
            img = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        else:
            features['image/img_path'] = tf.io.FixedLenFeature([], tf.string)
            x = tf.io.parse_single_example(tfrecord, features)
            image_encoded = tf.io.read_file(x['image/img_path'])
            img = tf.image.decode_jpeg(image_encoded, channels=3)

        labels = tf.stack(
            [tf.sparse.to_dense(x['image/object/bbox/xmin']),
             tf.sparse.to_dense(x['image/object/bbox/ymin']),
             tf.sparse.to_dense(x['image/object/bbox/xmax']),
             tf.sparse.to_dense(x['image/object/bbox/ymax']),
             tf.sparse.to_dense(x['image/object/landmark0/x']),
             tf.sparse.to_dense(x['image/object/landmark0/y']),
             tf.sparse.to_dense(x['image/object/landmark1/x']),
             tf.sparse.to_dense(x['image/object/landmark1/y']),
             tf.sparse.to_dense(x['image/object/landmark2/x']),
             tf.sparse.to_dense(x['image/object/landmark2/y']),
             tf.sparse.to_dense(x['image/object/landmark3/x']),
             tf.sparse.to_dense(x['image/object/landmark3/y']),
             tf.sparse.to_dense(x['image/object/landmark4/x']),
             tf.sparse.to_dense(x['image/object/landmark4/y']),
             tf.sparse.to_dense(x['image/object/landmark/valid'])], axis=1)

        img, labels = _transform_data(
            img_dim, using_flip, using_distort, using_encoding, priors,
            match_thresh, ignore_thresh, variances)(img, labels)

        return img, labels
    return parse_tfrecord


def _transform_data(img_dim, using_flip, using_distort, using_encoding, priors,
                    match_thresh, ignore_thresh, variances):
    def transform_data(img, labels):
        img = tf.cast(img, tf.float32)

        # randomly crop
        img, labels = _crop(img, labels)

        # padding to square
        img = _pad_to_square(img)

        # resize
        img, labels = _resize(img, labels, img_dim)

        # randomly left-right flip
        if using_flip:
            img, labels = _flip(img, labels)

        # distort
        if using_distort:
            img = _distort(img)

        # encode labels to feature targets
        if using_encoding:
            labels = encode_tf(labels=labels, priors=priors,
                               match_thresh=match_thresh,
                               ignore_thresh=ignore_thresh,
                               variances=variances)

        return img, labels
    return transform_data


def load_tfrecord_dataset(tfrecord_name, batch_size, img_dim,
                          using_bin=True, using_flip=True, using_distort=True,
                          using_encoding=True, priors=None, match_thresh=0.45,
                          ignore_thresh=0.3, variances=[0.1, 0.2],
                          shuffle=True, buffer_size=10240):
    """load dataset from tfrecord"""
    if not using_encoding:
        assert batch_size == 1  # dynamic data len when using_encoding
    else:
        assert priors is not None

    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(img_dim, using_bin, using_flip, using_distort,
                        using_encoding, priors, match_thresh, ignore_thresh,
                        variances),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


###############################################################################
#   Data Augmentation                                                         #
###############################################################################
def _flip(img, labels):
    flip_case = tf.random.uniform([], 0, 2, dtype=tf.int32)

    def flip_func():
        flip_img = tf.image.flip_left_right(img)
        flip_labels = tf.stack([1 - labels[:, 2],  labels[:, 1],
                                1 - labels[:, 0],  labels[:, 3],
                                1 - labels[:, 6],  labels[:, 7],
                                1 - labels[:, 4],  labels[:, 5],
                                1 - labels[:, 8],  labels[:, 9],
                                1 - labels[:, 12], labels[:, 13],
                                1 - labels[:, 10], labels[:, 11],
                                labels[:, 14]], axis=1)

        return flip_img, flip_labels

    img, labels = tf.case([(tf.equal(flip_case, 0), flip_func)],
                          default=lambda: (img, labels))

    return img, labels


def _crop(img, labels, max_loop=250):
    shape = tf.shape(img)

    def matrix_iof(a, b):
        """
        return iof of a and b, numpy version for data augenmentation
        """
        lt = tf.math.maximum(a[:, tf.newaxis, :2], b[:, :2])
        rb = tf.math.minimum(a[:, tf.newaxis, 2:], b[:, 2:])

        area_i = tf.math.reduce_prod(rb - lt, axis=2) * \
            tf.cast(tf.reduce_all(lt < rb, axis=2), tf.float32)
        area_a = tf.math.reduce_prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / tf.math.maximum(area_a[:, tf.newaxis], 1)

    def crop_loop_body(i, img, labels):
        valid_crop = tf.constant(1, tf.int32)

        pre_scale = tf.constant([0.3, 0.45, 0.6, 0.8, 1.0], dtype=tf.float32)
        scale = pre_scale[tf.random.uniform([], 0, 5, dtype=tf.int32)]
        short_side = tf.cast(tf.minimum(shape[0], shape[1]), tf.float32)
        h = w = tf.cast(scale * short_side, tf.int32)
        h_offset = tf.random.uniform([], 0, shape[0] - h + 1, dtype=tf.int32)
        w_offset = tf.random.uniform([], 0, shape[1] - w + 1, dtype=tf.int32)
        roi = tf.stack([w_offset, h_offset, w_offset + w, h_offset + h])
        roi = tf.cast(roi, tf.float32)

        value = matrix_iof(labels[:, :4], roi[tf.newaxis])
        valid_crop = tf.cond(tf.math.reduce_any(value >= 1),
                             lambda: valid_crop, lambda: 0)

        centers = (labels[:, :2] + labels[:, 2:4]) / 2
        mask_a = tf.reduce_all(
            tf.math.logical_and(roi[:2] < centers, centers < roi[2:]),
            axis=1)
        labels_t = tf.boolean_mask(labels, mask_a)
        valid_crop = tf.cond(tf.reduce_any(mask_a),
                             lambda: valid_crop, lambda: 0)

        img_t = img[h_offset:h_offset + h, w_offset:w_offset + w, :]
        h_offset = tf.cast(h_offset, tf.float32)
        w_offset = tf.cast(w_offset, tf.float32)
        labels_t = tf.stack(
            [labels_t[:, 0] - w_offset,  labels_t[:, 1] - h_offset,
             labels_t[:, 2] - w_offset,  labels_t[:, 3] - h_offset,
             labels_t[:, 4] - w_offset,  labels_t[:, 5] - h_offset,
             labels_t[:, 6] - w_offset,  labels_t[:, 7] - h_offset,
             labels_t[:, 8] - w_offset,  labels_t[:, 9] - h_offset,
             labels_t[:, 10] - w_offset, labels_t[:, 11] - h_offset,
             labels_t[:, 12] - w_offset, labels_t[:, 13] - h_offset,
             labels_t[:, 14]], axis=1)

        return tf.cond(valid_crop == 1,
                       lambda: (max_loop, img_t, labels_t),
                       lambda: (i + 1, img, labels))

    _, img, labels = tf.while_loop(
        lambda i, img, labels: tf.less(i, max_loop),
        crop_loop_body,
        [tf.constant(-1), img, labels],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None, 3]),
                          tf.TensorShape([None, 15])])

    return img, labels


def _pad_to_square(img):
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]

    def pad_h():
        img_pad_h = tf.ones([width - height, width, 3]) * \
            tf.reduce_mean(img, axis=[0, 1], keepdims=True)
        return tf.concat([img, img_pad_h], axis=0)

    def pad_w():
        img_pad_w = tf.ones([height, height - width, 3]) * \
            tf.reduce_mean(img, axis=[0, 1], keepdims=True)
        return tf.concat([img, img_pad_w], axis=1)

    img = tf.case([(tf.greater(height, width), pad_w),
                   (tf.less(height, width), pad_h)], default=lambda: img)

    return img


def _resize(img, labels, img_dim):
    w_f = tf.cast(tf.shape(img)[1], tf.float32)
    h_f = tf.cast(tf.shape(img)[0], tf.float32)
    locs = tf.stack([labels[:, 0] / w_f,  labels[:, 1] / h_f,
                     labels[:, 2] / w_f,  labels[:, 3] / h_f,
                     labels[:, 4] / w_f,  labels[:, 5] / h_f,
                     labels[:, 6] / w_f,  labels[:, 7] / h_f,
                     labels[:, 8] / w_f,  labels[:, 9] / h_f,
                     labels[:, 10] / w_f, labels[:, 11] / h_f,
                     labels[:, 12] / w_f, labels[:, 13] / h_f], axis=1)
    locs = tf.clip_by_value(locs, 0, 1)
    labels = tf.concat([locs, labels[:, 14][:, tf.newaxis]], axis=1)

    resize_case = tf.random.uniform([], 0, 5, dtype=tf.int32)

    def resize(method):
        def _resize():
            return tf.image.resize(
                img, [img_dim, img_dim], method=method, antialias=True)
        return _resize

    img = tf.case([(tf.equal(resize_case, 0), resize('bicubic')),
                   (tf.equal(resize_case, 1), resize('area')),
                   (tf.equal(resize_case, 2), resize('nearest')),
                   (tf.equal(resize_case, 3), resize('lanczos3'))],
                  default=resize('bilinear'))

    return img, labels


def _distort(img):
    img = tf.image.random_brightness(img, 0.4)
    img = tf.image.random_contrast(img, 0.5, 1.5)
    img = tf.image.random_saturation(img, 0.5, 1.5)
    img = tf.image.random_hue(img, 0.1)

    return img
