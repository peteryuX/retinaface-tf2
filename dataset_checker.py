import time
import cv2
import numpy as np
from modules.dataset import load_tfrecord_dataset
from modules.anchor import prior_box, decode_tf
from modules.utils import draw_bbox_landm, draw_anchor


using_bin = True
batch_size = 1
min_sizes = [[16, 32], [64, 128], [256, 512]]
steps = [8, 16, 32]
clip = False

img_dim = 640
priors = prior_box((img_dim, img_dim), min_sizes, steps, clip)

visualization = True  # False for time cost estimattion
using_encoding = True  # batch size should be 1 when False
variances = [0.1, 0.2]
match_thresh = 0.45
ignore_thresh = 0.3
num_samples = 100

if using_bin:
    tfrecord_name = './data/widerface_train_bin.tfrecord'
else:
    tfrecord_name = './data/widerface_train.tfrecord'

train_dataset = load_tfrecord_dataset(
    tfrecord_name, batch_size, img_dim=640, using_bin=using_bin,
    using_flip=True, using_distort=False, using_encoding=using_encoding,
    priors=priors, match_thresh=match_thresh, ignore_thresh=ignore_thresh,
    variances=variances, shuffle=False)

start_time = time.time()
for idx, (inputs, labels) in enumerate(train_dataset.take(num_samples)):
    print("{} inputs:".format(idx), inputs.shape, "labels:", labels.shape)

    if not visualization:
        continue

    img = np.clip(inputs.numpy()[0], 0, 255).astype(np.uint8)
    if not using_encoding:
        # labels includes loc, landm, landm_valid.
        targets = labels.numpy()[0]
        for target in targets:
            draw_bbox_landm(img, target, img_dim, img_dim)
    else:
        # labels includes loc, landm, landm_valid, conf.
        targets = decode_tf(labels[0], priors, variances=variances).numpy()
        for prior_index in range(len(targets)):
            if targets[prior_index][-1] == 1:
                draw_bbox_landm(img, targets[prior_index], img_dim, img_dim)
                draw_anchor(img, priors[prior_index], img_dim, img_dim)

    cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(0) == ord('q'):
        exit()

print("data fps: {:.2f}".format(num_samples / (time.time() - start_time)))
