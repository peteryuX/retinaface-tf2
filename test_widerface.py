from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf

from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)


flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('save_folder', './widerface_evaluate/widerface_txt/',
                    'folder path to save evaluate results')
flags.DEFINE_boolean('origin_size', True,
                     'whether use origin image size to evaluate')
flags.DEFINE_boolean('save_image', True, 'whether save evaluation images')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.02, 'score threshold for nms')
flags.DEFINE_float('vis_th', 0.5, 'threshold for visualization')


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


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    # evaluation on testing dataset
    testset_folder = cfg['testing_dataset_path']
    testset_list = os.path.join(testset_folder, 'label.txt')

    img_paths, _ = load_info(testset_list)
    for img_index, img_path in enumerate(img_paths):
        print(" [{} / {}] det {}".format(img_index + 1, len(img_paths),
                                         img_path))
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        # testing scale
        target_size = 1600
        max_size = 2150
        img_shape = img.shape
        img_size_min = np.min(img_shape[0:2])
        img_size_max = np.max(img_shape[0:2])
        resize = float(target_size) / float(img_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * img_size_max) > max_size:
            resize = float(max_size) / float(img_size_max)
        if FLAGS.origin_size:
            if os.path.basename(img_path) == '6_Funeral_Funeral_6_618.jpg':
                resize = 0.5 # this image is too big to avoid OOM problem
            else:
                resize = 1

        img = cv2.resize(img, None, None, fx=resize, fy=resize,
                         interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

        # run model
        outputs = model(img[np.newaxis, ...]).numpy()

        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        # write results
        img_name = os.path.basename(img_path)
        sub_dir = os.path.basename(os.path.dirname(img_path))
        save_name = os.path.join(
            FLAGS.save_folder, sub_dir, img_name.replace('.jpg', '.txt'))

        pathlib.Path(os.path.join(FLAGS.save_folder, sub_dir)).mkdir(
            parents=True, exist_ok=True)

        with open(save_name, "w") as file:
            bboxs = outputs[:, :4]
            confs = outputs[:, -1]

            file_name = img_name + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            file.write(file_name)
            file.write(bboxs_num)
            for box, conf in zip(bboxs, confs):
                x = int(box[0] * img_width_raw)
                y = int(box[1] * img_height_raw)
                w = int(box[2] * img_width_raw) - int(box[0] * img_width_raw)
                h = int(box[3] * img_height_raw) - int(box[1] * img_height_raw)
                confidence = str(conf)
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) \
                    + " " + confidence + " \n"
                file.write(line)

        # save images
        pathlib.Path(os.path.join(
            './results', cfg['sub_name'], sub_dir)).mkdir(
                parents=True, exist_ok=True)
        if FLAGS.save_image:
            for prior_index in range(len(outputs)):
                if outputs[prior_index][15] >= FLAGS.vis_th:
                    draw_bbox_landm(img_raw, outputs[prior_index],
                                    img_height_raw, img_width_raw)
            cv2.imwrite(os.path.join('./results', cfg['sub_name'], sub_dir,
                                     img_name), img_raw)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
