import tensorflow as tf
import menpo.io as mio
import time
import argparse
import glob
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'ShownX'

"""
This script is used to test mdm (Mnemonic Descent Method) to the images with bounding box

Input:
    -i: images folder
    -b: bounding box folder
    -o: output folder

Sample Use: python mdm_test.py -im ./images -bb ./bbox -o ./landmark

References: Mnemonic Descent Method: A recurrent process applied for end-to-end face alignment
G. Trigeorgis, P. Snape, M. A. Nicolaou, E. Antonakos, S. Zafeiriou.
Proceedings of IEEE International Conference on Computer Vision & Pattern Recognition (CVPR'16).
Las Vegas, NV, USA, June 2016.
"""

parser = argparse.ArgumentParser(description="MDM input")
parser.add_argument('-i', '--image', help='Image folder', required=True)
parser.add_argument('-ie', '--im_ext', help='Image extension', required=False)
parser.add_argument('-b', '--bbox', help='Bounding box folder', required=True)
parser.add_argument('-be', '--bb_ext', help='Bounding box extension', required=False)
parser.add_argument('-o', '--output', help='Output folder', required=True)
parser.add_argument('-m', '--model', help='Model path', required=True)

args = parser.parse_args()

if not args.im_ext:
    args.im_ext = 'jpg'

if not args.bb_ext:
    args.bb_ext = 'roi'

print("Image folder: %s" % args.image)
print("Image extension: %s" % args.im_ext)
print("Bounding box folder: %s" % args.bbox)
print("Bounding box extension: %s" % args.bb_ext)
print("Output folder: %s" % args.output)
print("Model path is: %s" % args.model)


def read_bbox(bbox_path):
    """
    Read bounding box from the file according to the bbox path which contains the bounding box like (x, y, w, h)
    Then convert it to (y1, x1, y2, x2)
    Args:
        bbox_path: bounding box path
    Returns:
        bbox: bounding box
    """
    infile = open(bbox_path, 'r')
    line = infile.readline()
    values = map(float, line.split(','))
    # change (x, y, w, h) --> (y1, x1, y2, x2)
    bbox = np.array([values[1], values[0], values[1] + values[3], values[0] + values[2]])
    infile.close()
    return bbox


def read_bbox2(bbox_path):
    """
    Read bounding box from the file according to the bbox path which contains the bounding box like (x1, y1, x2, y2)
    Args:
        bbox_path: bounding box path
    Returns:
        bbox: bounding box
    """
    infile = open(bbox_path, 'r')
    line = infile.readline()
    values = map(float, line.split(','))
    # keep (x, y, w, h)
    bbox = np.array([values[0], values[1], values[2], values[3]])
    infile.close()
    return bbox


def write_lm(lm_path, pts):
    """
    write landmarks to the file
    Args:
        lm_path: the landmark file path
        pts: prediction landmarks
    """
    outfile = open(lm_path, 'w+')
    for pt in pts:
        outfile.write('%d %d\n' % (pt[1], pt[0]))
    outfile.close()


# glob the images from the image folder
imgs_path = op.join(args.image, '*.'+args.im_ext)
img_list = glob.glob(imgs_path)


# the image to fit (rgb image of HWC) where H: height, W: weight and C
# the number of channels (=3).
image = tf.placeholder(tf.float32, shape=(None, None, 3), name='images')
# we only use the upper-left (x0, y0) and lower-down (x1, y1) points
# of the bounding box as a vector (x0, y0, x1, y1).
initial_bb = tf.placeholder(tf.float32, shape=(4), name='inits')

MDM_MODEL_PATH = args.model

with open(MDM_MODEL_PATH, 'rb') as f:
    graph_def = tf.GraphDef.FromString(f.read())
    pred,  = tf.import_graph_def(graph_def, input_map={"image": image, "bounding_box": initial_bb}, return_elements=['prediction:0'])

sess = tf.Session()

start_time = time.time()

SHOW = False

for img_path in img_list:
    print(img_path)
    img_name = op.basename(img_path)
    bbx_name = img_name.replace(args.im_ext, args.bb_ext)
    bbx_path = op.join(args.bbox, bbx_name)
    # Read image
    im = mio.import_image(img_path)
    # Read bounding box
    if op.exists(bbx_path):
        boundingbox = read_bbox(bbx_path)
        #im.landmarks['bb'] = mio.import_landmark_file(bbx_path)
        #boundingbox = im.landmarks['bb'].lms.bounding_box()
        # Predict landmarks
        prediction, = sess.run(pred, feed_dict={
            # menpo stores images CHW instead of HWC that tensorflow uses
            image: im.pixels.transpose(1, 2, 0),
            # grab the upper-left and lower-down points of the bounding box.
            initial_bb: boundingbox})
        prediction = prediction.astype(int)
        # Write to the file
        pts_path = op.join(args.output, img_name.replace(args.im_ext, 'lm2'))
        write_lm(pts_path, prediction)
        if SHOW:
            im = plt.imread(img_path)
            plt.axis('off')
            plt.imshow(im)
            plt.plot(prediction[:, 1], prediction[:, 0], 'ro')
            plt.show()

elapsed_time = time.time() - start_time
print('The total time elapsed is: %.2f s, average time is: %.2f s/image' %(elapsed_time, elapsed_time/len(img_list)))