import torch
import torchvision
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import os
import csv


def read_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return np.asarray(img)


def get_simple_image_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)


def create_grabcut_mask(image, grabcut_mask):
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    grabcut_mask, _, _ = cv2.grabCut(
        image,
        grabcut_mask,
        None,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )
    return np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype(np.uint8)


class ForeGroundExtractor:
    def __init__(self, mrcnn_pre_process,
                 mrcnn_confidence=0.2,
                 grabcut_foreground_confidence=0.8,
                 detect_object_labels=[1, 74]):
        self.mrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.mrcnn.eval()
        self.mrcnn_confidence = mrcnn_confidence
        self.grabcut_foreground = grabcut_foreground_confidence
        self.trans = mrcnn_pre_process
        self.detect_object_labels = detect_object_labels

    def mrcnn_output2grabcut_input(self, output):
        boxes = output[0]['boxes'].detach().numpy()
        masks = output[0]['masks'].detach().numpy()
        labels = output[0]['labels'].detach().numpy()
        scores = output[0]['scores'].detach().numpy()
        grab_mask = np.zeros(masks.shape[2:], np.uint8)
        for label in self.detect_object_labels:
            boxes_label = boxes[(self.mrcnn_confidence < scores) & (labels == label)].astype(np.uint64)
            masks_label = masks[(self.mrcnn_confidence < scores) & (labels == label)]
            for b in boxes_label:
                grab_mask[b[1]:b[3]:, b[0]:b[2]] = cv2.GC_PR_BGD
            for m in masks_label:
                grab_mask[self.grabcut_foreground < m[0]] = cv2.GC_FGD
        return grab_mask

    def detect_foreground(self, image):
        output = self.mrcnn([self.trans(Image.fromarray(image))])
        grabcut_input = self.mrcnn_output2grabcut_input(output)
        if not (grabcut_input == cv2.GC_FGD).any():
            return np.zeros(image.shape[:2]).astype(np.uint8)
        return create_grabcut_mask(image, grabcut_input)


fge = ForeGroundExtractor(get_simple_image_transform())


def CalculateFg2BgRatio(SubDirectory, cache_save_path, csv_save_path):
    result = []
    cache_folder_name = os.path.basename(os.path.normpath(SubDirectory))
    cache_path = os.path.join(cache_save_path, cache_folder_name)
    os.makedirs(cache_path, exist_ok=True)

    for file_pic in os.listdir(SubDirectory):
        if file_pic.endswith('.jpg'):
            pic_dect = read_image(os.path.join(SubDirectory, file_pic))
            foreground_mask = fge.detect_foreground(pic_dect)
            image_show = Image.fromarray(255 * foreground_mask)
            image_show.save(os.path.join(cache_path, file_pic))

    for pic in os.listdir(cache_path):
        if pic.endswith('.jpg'):
            pic = os.path.join(cache_path, pic)
            image = cv2.imread(pic, cv2.IMREAD_UNCHANGED)
            number_of_white_pix = np.sum(image == 255)
            number_of_black_pix = np.sum(image == 0)
            number_of_total = number_of_white_pix + number_of_black_pix
            number_of_ratio = number_of_white_pix / number_of_black_pix
            result.append([pic, number_of_white_pix, number_of_black_pix, number_of_total, number_of_ratio])

    csv_filename = os.path.join(csv_save_path, cache_folder_name + "_Fg2BgRatio" + ".csv")
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image",
                         "Number of White Pixels",
                         "Number of Black Pixels",
                         "Number of Total Pixels",
                         "Ratio of White to Black Pixels"])
        writer.writerows(result)
