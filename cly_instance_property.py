import numpy as np
from scipy import io
import cv2 as cv
from tqdm import tqdm
import os
from PIL import Image
from pathlib import Path
import math


def get_contour_from_segmentation(inst_seg):
    """
    :param inst_seg: the segmentation of instance
    :return: (int) the number of instance boundary pixels
    """
    if inst_seg.dtype != 'uint8':
        inst_seg = inst_seg.astype(np.uint8)
    binary_seg = np.zeros_like(inst_seg)
    # augment
    binary_seg[inst_seg != 0] = 255
    # =========== ????
    binary_seg = binary_seg.copy()
    if cv.__version__ < '4.0':
        binary_seg, contours, hierarchy = cv.findContours(binary_seg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv.findContours(binary_seg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.imwrite('imgs/full.png', binary_seg)
    contour_img = np.zeros_like(binary_seg)
    cv.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
    # cv.imwrite('imgs/contour.png', contour_img)
    return (contour_img != 0).sum()


class VOC2012:
    def __init__(self, dataset_path):
        self.index = 0
        self.dataset_path = dataset_path
        with open('{}/ImageSets/Segmentation/val.txt'.format(self.dataset_path)) as f:
            val_img_ids = sorted(f.read().splitlines())

        self.src_imgs, self.inst_segs, self.inst_ppts = [], [], []

        print('loading VOC2012')
        for val_img_id in tqdm(val_img_ids):
            class_label = np.array(Image.open('{}/SegmentationClass/{}.png'.format(self.dataset_path, val_img_id)))
            instance_label = np.array(Image.open('{}/SegmentationObject/{}.png'.format(self.dataset_path, val_img_id)))
            # class_label[class_label == 255] = 0
            # instance_label[instance_label == 255] = 0

            # for each instance in image
            inst_ids = sorted(set(instance_label.flat) - {0, 255})
            for i in inst_ids:
                # src_img
                img = np.array(Image.open('{}/JPEGImages/{}.jpg'.format(self.dataset_path, val_img_id)))

                # inst_seg
                seg = np.zeros_like(instance_label)
                seg[instance_label == i] = 1

                # ppt: property including class_id, size, contour and compactness
                remain_ids = list(set(np.unique(class_label * seg)) - {0})
                if len(remain_ids) != 1:
                    raise Exception('image exists multi-classes.')
                cls_id = remain_ids[0]
                size = seg.sum()
                contour = get_contour_from_segmentation(seg)
                ppt = [cls_id, size, contour, 4 * math.pi * size / contour / contour]

                # append
                self.src_imgs.append(img)
                seg[instance_label == 255] = 255
                self.inst_segs.append(seg)
                self.inst_ppts.append(ppt)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.src_imgs)

    def __next__(self):
        if self.index > len(self) - 1:
            raise StopIteration
        result = (self.src_imgs[self.index], self.inst_segs[self.index], self.inst_ppts[self.index])
        self.index += 1
        return result


class miniCOCO:
    def __init__(self, dataset_path):
        self.cls_index = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']
        self.index = 0
        self.dataset_path = dataset_path
        self.src_imgs, self.inst_segs, self.inst_ppts = [], [], []
        img_names = sorted(os.listdir('{}/img/'.format(self.dataset_path)))
        print('loading miniCOCO')
        for img_name in tqdm(img_names):
            # src img
            # =============== option 1: PIL ==============
            # img = np.array(Image.open('{}/img/{}'.format(self.dataset_path, img_name)))
            # if len(img.shape) == 2:   # for gray image
            #     img = np.expand_dims(img, axis=2).repeat(3, axis=2)
            # ================== option 2: opencv =================
            import cv2
            img = cv2.imread('{}/img/{}'.format(self.dataset_path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # ===================== the results would be with slight differences =========

            # inst_seg
            # ============= option 1: PIL ===================
            # seg = np.array(Image.open('{}/gt/{}'.format(self.dataset_path, img_name.replace('jpg', 'png'))))
            # seg[seg == 255] = 1
            # ============= option 2: opencv =================
            seg_path = '{}/gt/{}'.format(self.dataset_path, img_name.replace('jpg', 'png'))
            seg = np.max(cv2.imread(seg_path).astype(np.int32), axis=2)
            seg[seg > 0] = 1
            # ===============================================================================

            # ppt
            cls_name = img_name.split('_')[0][:-2]
            cls_id = self.cls_index.index(cls_name)
            size = seg.sum()
            contour = get_contour_from_segmentation(seg)
            ppt = [cls_id, size, contour, 4 * math.pi * size / contour / contour]

            # append
            self.src_imgs.append(img)
            self.inst_segs.append(seg)
            self.inst_ppts.append(ppt)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.src_imgs)

    def __next__(self):
        if self.index > len(self) - 1:
            raise StopIteration
        result = (self.src_imgs[self.index], self.inst_segs[self.index], self.inst_ppts[self.index])
        self.index += 1
        return result


class COCO2017:
    # from pycocotools.coco import COCO
    def __init__(self, dataset_path):
        self.index = 0
        self.dataset_path = dataset_path
        coco_inst_path = '{}/annotations/instances_val2017.json'.format(dataset_path)
        self.coco = COCO(coco_inst_path)
        val_inst_ids = self.coco.getAnnIds()
        self.val_anns = self.coco.loadAnns(val_inst_ids)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.val_anns)

    def __next__(self):
        if self.index > len(self) - 1: raise StopIteration
        annotation = self.val_anns[self.index]
        # src img
        img_info = self.coco.loadImgs(annotation['image_id'])[0]
        src_img_name = img_info['file_name']
        src_img = np.array(Image.open('{}/val2017/{}'.format(self.dataset_path, src_img_name)))
        if len(src_img.shape) == 2:
            src_img = np.expand_dims(src_img, axis=2).repeat(3, axis=2)

        # inst_seg
        seg = self.coco.annToMask(annotation)

        # ppt
        cls_id = annotation['category_id']
        size = int(annotation['area'])
        contour = get_contour_from_segmentation(seg)
        ppt = [cls_id, size, contour, 4 * math.pi * size / contour / contour]
        result = (src_img, seg, ppt)
        self.index += 1
        return result


class SBD:
    def __init__(self, dataset_path):
        self.index = 0
        self.dataset_path = dataset_path
        with open('{}/val.txt'.format(self.dataset_path)) as f:
            val_img_ids = sorted(f.read().splitlines())

        print('Preprocessing!')
        self.src_imgs, self.inst_segs, self.inst_ppts = [], [], []


        print('loading SBD')
        for val_img_id in tqdm(val_img_ids):
            mat = io.loadmat('{}/inst/{}.mat'.format(self.dataset_path, val_img_id))
            inst_cnt = len(mat['GTinst'][0]['Categories'][0])
            category = [mat['GTinst'][0]['Categories'][0][i][0] for i in range(inst_cnt)]
            full_seg = mat['GTinst'][0]['Segmentation'][0]
            boundary = [mat['GTinst'][0]['Boundaries'][0][i][0].toarray() for i in range(inst_cnt)]
            for i in range(inst_cnt):
                # src_img
                img = np.array(Image.open('{}/img/{}.jpg'.format(self.dataset_path, val_img_id)))

                # inst_seg
                inst_seg = np.zeros_like(full_seg)
                inst_seg[full_seg == i + 1] = 1

                # ppt: property including class_id, size, contour and compactness
                cls_id = category[i]
                size = inst_seg.sum()
                contour = boundary[i].sum()
                ppt = [cls_id, size, contour, 4 * math.pi * size / contour / contour]

                # append
                self.src_imgs.append(img)
                self.inst_segs.append(inst_seg)
                self.inst_ppts.append(ppt)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.src_imgs)

    def __next__(self):
        if self.index > len(self) - 1:
            raise StopIteration
        result = (self.src_imgs[self.index], self.inst_segs[self.index], self.inst_ppts[self.index])
        self.index += 1
        return result


if __name__ == "__main__":
    voc_path = 'C:/Users/49093/Desktop/ra_seg/dataset/VOC2012'
    coco_path = 'C:/Users/49093/Desktop/ra_seg/dataset/COCO'
    mcoco_path = 'C:/Users/49093/Desktop/ra_seg/dataset/COCO_MVal'
    sbd_path = 'C:/Users/49093/Desktop/ra_seg/dataset/SBD'
    # voc = VOC2012(voc_path)
    # coco = COCO2017(coco_path)
    cocomini = miniCOCO(mcoco_path)
