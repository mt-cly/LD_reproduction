from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter.ttk import Frame, Style
import os, time, cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import tkinter.messagebox as mbox
from our_func_cvpr18 import cly_our_func, our_func, build
from cly_instance_property import miniCOCO, SBD
from tqdm import tqdm
import tensorflow as tf


def get_next_anno_point(pred, gt, seq_points):
    fndist_map = distance_transform_edt(np.pad((gt == 1) & (pred == 0), ((1, 1), (1, 1)), 'constant'))[1:-1, 1:-1]
    fpdist_map = distance_transform_edt(np.pad((gt == 0) & (pred == 1), ((1, 1), (1, 1)), 'constant'))[1:-1, 1:-1]
    fndist_map[seq_points[:, 1], seq_points[:, 0]], fpdist_map[seq_points[:, 1], seq_points[:, 0]] = 0, 0
    [usr_map, if_pos] = [fndist_map, 1] if fndist_map.max() > fpdist_map.max() else [fpdist_map, 0]
    [y_mlist, x_mlist] = np.where(usr_map == usr_map.max())
    pt_next = (x_mlist[0], y_mlist[0], if_pos)
    return pt_next
    
def get_next_anno_point_prob(pred, gt, seq_points):
    fndist_map = distance_transform_edt(np.pad((gt == 1) & (pred == 0), ((1, 1), (1, 1)), 'constant'))[1:-1, 1:-1]
    fpdist_map = distance_transform_edt(np.pad((gt == 0) & (pred == 1), ((1, 1), (1, 1)), 'constant'))[1:-1, 1:-1]
    fndist_map[seq_points[:, 1], seq_points[:, 0]], fpdist_map[seq_points[:, 1], seq_points[:, 0]] = 0, 0
    neg_prob_map, pos_prob_map = fndist_map.reshape(-1), fpdist_map.reshape(-1)
    prob_map = np.max((neg_prob_map, pos_prob_map), axis=0)
    prob_map = prob_map/prob_map.sum()
    pnt_index = np.random.choice(len(prob_map), replace=False, p=prob_map)
    if_pos = 1 if neg_prob_map[pnt_index]>0 else 0 if pos_prob_map[pnt_index]>0 else -1
    assert if_pos != -1
    h, w = fndist_map.shape
    pt_next = (pnt_index%w, int(pnt_index/w), if_pos)
    return pt_next

def eval_dataset(dataset_name, dataset_iter, max_point_num=20, record_point_num=20, miou_target=0.85):
    # ===================== load model =============================
    sess = tf.Session()
    input = tf.placeholder(tf.float32, shape=[None, None, None, 7])
    sz = tf.placeholder(tf.int32, shape=[2])
    network = build(input, sz)
    saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state("Models/ours_cvpr18")
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # =============================================================

    instance_info = []
    for img, gt, ppt in tqdm(dataset_iter):
        pred = np.zeros_like(gt)
        seq_points = np.empty([0, 3], dtype=np.int64)
        if_get_target = False

        # prepare for initial input
        int_pos = np.uint8(255 * np.ones(img.shape[:2]))
        int_neg = np.uint8(255 * np.ones(img.shape[:2]))

        # simulate the click processing
        NoC, mIoU_NoC = 0, [0] * (record_point_num + 1)

        for point_num in range(1, max_point_num + 1):
            # ================ get next click through argmax =============
            # pt_next = get_next_anno_point(pred, gt, seq_points)
            # ================= baseline: get next click through probability distribution ============
            pt_next = get_next_anno_point_prob(pred, gt, seq_points)
            
            seq_points = np.append(seq_points, [pt_next], axis=0)
            
            # ================ replace the model ==============
            pred, int_pos, int_neg = cly_our_func((sess, network, input, sz), img, int_pos, int_neg, seq_points)
            # =================================================
            miou = ((pred == 1) & (gt == 1)).sum() / (((pred == 1) | (gt == 1)) & (gt != 255)).sum()
            if point_num <= record_point_num:
                mIoU_NoC[point_num] += miou
            if (not if_get_target) and (miou >= miou_target or point_num == max_point_num):
                NoC += point_num
                if_get_target = True
            if if_get_target and point_num >= record_point_num:
                break
        print(mIoU_NoC, NoC)    
        # save info for each instance
        instance_info.append([ppt, [NoC], mIoU_NoC])
    np.save('ld_{}val_baseline'.format(dataset_name), np.array(instance_info))


def main():
    dataset_path = '/home/liyi/datasets'
    
    # coco_dataset_iter = miniCOCO('{}/{}'.format(dataset_path, 'COCO_MVal'))
    # eval_dataset('COCO_MVal', coco_dataset_iter)

    sbd_dataset_iter = SBD('{}/{}'.format(dataset_path, 'SBD'))
    eval_dataset('SBD', sbd_dataset_iter)


if __name__ == '__main__':
    main()
