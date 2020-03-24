from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import random
import tensorflow as tf
import numpy as np


flags.DEFINE_string('dataset_path', './data/widerface/train',
                    'path to dataset')
flags.DEFINE_string('output_path', './data/widerface_train_bin.tfrecord',
                    'path to ouput tfrecord')
flags.DEFINE_boolean('is_binary', True, 'whether save images as binary files'
                     ' or load them on the fly.')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def make_example(img_name, img_path, target, is_binary):
    # Create a dictionary with features that may be relevant.
    feature = {'image/img_name': _bytes_feature([img_name]),
               'image/object/bbox/xmin': _float_feature(target[:, 0]),
               'image/object/bbox/ymin': _float_feature(target[:, 1]),
               'image/object/bbox/xmax': _float_feature(target[:, 2]),
               'image/object/bbox/ymax': _float_feature(target[:, 3]),
               'image/object/landmark0/x': _float_feature(target[:, 4]),
               'image/object/landmark0/y': _float_feature(target[:, 5]),
               'image/object/landmark1/x': _float_feature(target[:, 6]),
               'image/object/landmark1/y': _float_feature(target[:, 7]),
               'image/object/landmark2/x': _float_feature(target[:, 8]),
               'image/object/landmark2/y': _float_feature(target[:, 9]),
               'image/object/landmark3/x': _float_feature(target[:, 10]),
               'image/object/landmark3/y': _float_feature(target[:, 11]),
               'image/object/landmark4/x': _float_feature(target[:, 12]),
               'image/object/landmark4/y': _float_feature(target[:, 13]),
               'image/object/landmark/valid': _float_feature(target[:, 14])}
    if is_binary:
        img_str = open(img_path, 'rb').read()
        feature['image/encoded'] = _bytes_feature([img_str])
    else:
        feature['image/img_path'] = _bytes_feature([img_path])

    return tf.train.Example(features=tf.train.Features(feature=feature))


def load_info(txt_path):
    """load info from txt"""
    img_paths = []
    words = []

    f = open(txt_path, 'r')
    lines = f.readlines()
    isFirst = True
    labels = []
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            if isFirst is True:
                isFirst = False
            else:
                labels_copy = labels.copy()
                words.append(labels_copy)
                labels.clear()
            path = line[2:]
            path = txt_path.replace('label.txt', 'images/') + path
            img_paths.append(path)
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            labels.append(label)

    words.append(labels)
    return img_paths, words


def get_target(labels):
    annotations = np.zeros((0, 15))
    if len(labels) == 0:
        return annotations
    for idx, label in enumerate(labels):
        annotation = np.zeros((1, 15))
        # bbox
        annotation[0, 0] = label[0]  # x1
        annotation[0, 1] = label[1]  # y1
        annotation[0, 2] = label[0] + label[2]  # x2
        annotation[0, 3] = label[1] + label[3]  # y2

        # landmarks
        annotation[0, 4] = label[4]    # l0_x
        annotation[0, 5] = label[5]    # l0_y
        annotation[0, 6] = label[7]    # l1_x
        annotation[0, 7] = label[8]    # l1_y
        annotation[0, 8] = label[10]   # l2_x
        annotation[0, 9] = label[11]   # l2_y
        annotation[0, 10] = label[13]  # l3_x
        annotation[0, 11] = label[14]  # l3_y
        annotation[0, 12] = label[16]  # l4_x
        annotation[0, 13] = label[17]  # l4_y
        if (annotation[0, 4] < 0):
            annotation[0, 14] = -1  # w/o landmark
        else:
            annotation[0, 14] = 1

        annotations = np.append(annotations, annotation, axis=0)
    target = np.array(annotations)

    return target


def main(_):
    dataset_path = FLAGS.dataset_path

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    logging.info('Reading data list...')
    img_paths, words = load_info(os.path.join(dataset_path, 'label.txt'))
    samples = list(zip(img_paths, words))
    random.shuffle(samples)

    if os.path.exists(FLAGS.output_path):
        logging.info('{:s} already exists. Exit...'.format(
            FLAGS.output_path))
        exit()

    logging.info('Writing {} sample to tfrecord file...'.format(len(samples)))
    with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
        for img_path, word in tqdm.tqdm(samples):
            target = get_target(word)
            img_name = os.path.basename(img_path).replace('.jpg', '')

            tf_example = make_example(img_name=str.encode(img_name),
                                      img_path=str.encode(img_path),
                                      target=target,
                                      is_binary=FLAGS.is_binary)

            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
