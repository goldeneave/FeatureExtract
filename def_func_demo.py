"""

│Root
├──Directory 1 (Posts)
├────Sub Directory 1 (Brand 1)
├────Sub Directory 2 (Brand 2)
├────Sub Directory 3 (Brand 3)
│
├──Directory 2 (GV Results)
├────Sub Directory 1 (Brand 1)
├────Sub Directory 2 (Brand 2)
├────Sub Directory 3 (Brand 3)

"""

import os
import cv2
import csv
import json
import lzma
import math
import numpy as np
import pandas as pd
from PIL import Image
import torchvision
import tensorflow as tf
import torchvision.transforms as T


def ExtractEngageNumber(SubDirectory, output_path):
    result = []
    for file in os.listdir(SubDirectory):
        if file.endswith(".xz"):
            with lzma.open(os.path.join(SubDirectory, file), 'r') as f:
                data = json.load(f)

            try:
                node = data['node']
                edge_media_preview_like = node['edge_media_preview_like']
                edge_media_to_comment = node['edge_media_to_comment']
                count_like = edge_media_preview_like['count']
                count_comment = edge_media_to_comment['count']

            except KeyError:
                count_like = 0
                count_comment = 0

            result.append([file, count_comment, count_like])

    subdir_name = os.path.basename(os.path.normpath(SubDirectory))
    output_file = os.path.join(output_path, f"{subdir_name}_EngNum.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Comment Count", "Like Count"])
        writer.writerows(result)


def CountFaceAppear(SubDirectory, output_path):
    threshold = 0.3
    result = []

    for filename in os.listdir(SubDirectory):
        if filename.endswith('.json'):
            with open(os.path.join(SubDirectory, filename)) as file:
                data = json.load(file)
                count = 0
                for face in data["face_annotations"]:
                  if "detection_confidence" in face and face['detection_confidence'] > threshold:
                    count += 1
                # for item in data:
                #     if 'detection_confidence' in item and item['detection_confidence'] > threshold:
                #         count += 1
                result.append([filename, count])

    base_name = os.path.basename(SubDirectory)
    output_file = os.path.join(output_path, f"{base_name}_FaceAppear.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['File Name', 'Count'])
        writer.writerows(result)


def ImageColorfulPotential(SubDirectory, output_path):
    result = []
    for filename in os.listdir(SubDirectory):
        if filename.endswith(".json"):
            with open(os.path.join(SubDirectory, filename), 'r') as f:
                data = json.load(f)
                colors = data['image_properties_annotation']['dominant_colors']['colors']
                fraction = sum([color['pixel_fraction'] for color in sorted(colors, key=lambda x: -x['pixel_fraction'])])
                result.append({'file_name': filename, 'pixel_fraction': fraction})

    base_name = os.path.basename(SubDirectory)
    output_file = os.path.join(output_path, f"{base_name}_ColorPotential.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['file_name', 'pixel_fraction'])
        writer.writeheader()
        writer.writerows(result)


def HSVSpaceFeature(SubDirectory, output_path):
    image_files = [f for f in os.listdir(SubDirectory) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

    result = []
    for image_file in image_files:
        img_path = os.path.join(SubDirectory, image_file)
        img = cv2.imread(img_path)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        h, s, v = cv2.split(hsv_img)
        l, a, b = cv2.split(lab_img)

        average_brightness = np.mean(v)
        average_saturation = np.mean(s)
        average_hue = np.mean(h)
        lightness_std = np.std(l)

        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([40, 255, 255])

        red_mask = cv2.inRange(hsv_img, red_lower, red_upper)
        yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)

        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        height, width, _ = img.shape
        overall_pixels = height * width
        warm_color_pixels = red_pixels + yellow_pixels
        warm_ratio = warm_color_pixels / overall_pixels

        result.append([image_file, average_brightness, average_saturation, average_hue, lightness_std, warm_ratio])

    df = pd.DataFrame(result, columns=['file_name',
                                       'average_brightness',
                                       'average_saturation',
                                       'average_hue',
                                       'lightness_std',
                                       'warm_ratio'])
    df.to_csv(os.path.join(output_path, SubDirectory.split('/')[-1] + '_HSVFeature'+ '.csv'), index=False)


'''---------------------------下面是摘自GitHub的GrabCut算法，用于分离Fg和Bg--------------------------'''
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
                 detect_object_label=1):
        self.mrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.mrcnn.eval()
        self.mrcnn_confidence = mrcnn_confidence
        self.grabcut_foreground = grabcut_foreground_confidence
        self.trans = mrcnn_pre_process
        self.detect_object_label = detect_object_label

    def mrcnn_output2grabcut_input(self, output):
        boxes = output[0]['boxes'].detach().numpy()
        masks = output[0]['masks'].detach().numpy()
        labels = output[0]['labels'].detach().numpy()
        scores = output[0]['scores'].detach().numpy()
        boxes = boxes[(self.mrcnn_confidence < scores) & (labels == self.detect_object_label)].astype(np.uint64)
        masks = masks[(self.mrcnn_confidence < scores) & (labels == self.detect_object_label)]

        grab_mask = np.zeros(masks.shape[2:], np.uint8)
        for b in boxes:
            grab_mask[b[1]:b[3]:, b[0]:b[2]] = cv2.GC_PR_BGD
        for m in masks:
            grab_mask[self.grabcut_foreground < m[0]] = cv2.GC_FGD
        return grab_mask

    def detect_foreground(self, image):
        output = self.mrcnn([self.trans(Image.fromarray(image))])
        grabcut_input = self.mrcnn_output2grabcut_input(output)
        if not (grabcut_input == cv2.GC_FGD).any():
            return np.zeros(image.shape[:2]).astype(np.uint8)
        return create_grabcut_mask(image, grabcut_input)


'''------------------------加载Mask RCNN预训练模型，以上是摘自GitHub的GrabCut算法--------------------------'''
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


model = tf.keras.models.load_model('colormodel_trained_90.h5')
color_dict={
    0 : 'Red',
    1 : 'Green',
    2 : 'Blue',
    3 : 'Yellow',
    4 : 'Orange',
    5 : 'Pink',
    6 : 'Purple',
    7 : 'Brown',
    8 : 'Grey',
    9 : 'Black',
    10 : 'White'
}

def ColorPrediction(Red, Green, Blue):
  rgb = np.asarray((Red, Green, Blue))
  input_rgb = np.reshape(rgb, (-1,3))
  color_class_confidence = model.predict(input_rgb, verbose=0)
  color_index = np.argmax(color_class_confidence, axis=1)
  color = color_dict[int(color_index)]
  return color


def ColorPercentage(SubDirectory, output_dir):
  folder_name = os.path.basename(os.path.normpath(SubDirectory))
  output_path = os.path.join(output_dir, f'{folder_name}_ColorPencent.csv')

  for file in os.listdir(SubDirectory):
    if file.endswith('.json'):
      with open(os.path.join(SubDirectory, file), 'r') as f:
        data = json.load(f)
      colors = data['image_properties_annotation']['dominant_colors']['colors']
      color_groups = {}

      for color in colors:
        color_info = color['color']
        pixel_fraction = color['pixel_fraction']

        Red = color_info['red']
        Green = color_info['green']
        Blue = color_info['blue']
        result = ColorPrediction(int(Red), int(Green), int(Blue))

        if result not in color_groups:
          color_groups[result] = []
        color_groups[result].append(pixel_fraction)

      # sorted_color_groups = sorted(color_groups.items(), key=lambda x: x[1], reverse=True)

      color_totals = {color: 0 for color in color_dict.values()}

      for color, fractions in color_groups.items():
        color_totals[color] += sum(fractions)

      output = []
      for color, total in color_totals.items():
        output.append('{:,}'.format(total))

      with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.stat(output_path).st_size == 0:
          writer.writerow(['file_name'] + list(color_dict.values()))
        writer.writerow([file] + output)


def Face2Diagonal(Gjson_path, post_dir, output_dir):
    json_dir = Gjson_path
    image_dir = os.path.join(post_dir, os.path.basename(json_dir))
    folder_name = os.path.basename(os.path.normpath(Gjson_path))
    output_path = os.path.join(output_dir, f'{folder_name}_Face2Diag.csv')

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Json Filename", "Image Filename", "Distance"])

        for json_filename in os.listdir(json_dir):
          if not json_filename.endswith('.json'):
            continue

          json_file = os.path.join(json_dir, json_filename)
          with open(json_file, 'r') as f:
              data = json.load(f)

          face_annotations = data.get('face_annotations', [])
          if not face_annotations or 'detection_confidence' not in face_annotations[0]:
              continue

          image_filename = json_filename[:-5] + '.jpg'
          image_file = os.path.join(image_dir, image_filename)
          image = Image.open(image_file)
          image_width, image_height = image.size
          bounding_boxes = []
          for face in face_annotations:
              bounding_boxes.append(face['bounding_poly']['vertices'])

          areas = []
          for box in bounding_boxes:
              width = box[1]['x'] - box[0]['x']
              height = box[2]['y'] - box[0]['y']
              area = width * height
              areas.append(area)
          largest_box_index = areas.index(max(areas))
          largest_bounding_box = bounding_boxes[largest_box_index]

          x = (largest_bounding_box[0]['x'] + largest_bounding_box[1]['x']) / 2
          y = (largest_bounding_box[0]['y'] + largest_bounding_box[2]['y']) / 2
          center_point = (x, y)

          k1 = image_height / image_width
          b1 = 0
          k2 = -k1
          b2 = image_height

          point_x, point_y = center_point

          k_perp_1 = -1 / k1
          b_perp_1 = point_y - point_x * k_perp_1
          k_perp_2 = -1 / k2
          b_perp_2 = point_y - point_x * k_perp_2

          x_f1 = (b_perp_1 - b1) / (k1 - k_perp_1)
          y_f1 = k1 * x_f1 + b1
          x_f2 = (b_perp_2 - b2)
          y_f2 = k2 * x_f2 + b2

          distance_ab = math.sqrt((x_f1 - point_x) ** 2 + (y_f1 - point_y) ** 2)
          distance_ac = math.sqrt((x_f2 - point_x) ** 2 + (y_f2 - point_y) ** 2)
          distance = min(distance_ab, distance_ac)

          writer.writerow([json_filename, image_filename, distance])


def Face2IntersectionPoint(Gjson_path, post_dir, output_dir):
    json_dir = Gjson_path
    image_dir = os.path.join(post_dir, os.path.basename(json_dir))
    folder_name = os.path.basename(os.path.normpath(Gjson_path))
    output_path = os.path.join(output_dir, f'{folder_name}_Face2Point.csv')

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Json Filename", "Image Filename", "Distance"])

        for json_filename in os.listdir(json_dir):
            if not json_filename.endswith('.json'):
                continue

            json_file = os.path.join(json_dir, json_filename)
            with open(json_file, 'r') as f:
                data = json.load(f)

            face_annotations = data.get('face_annotations', [])
            if not face_annotations or 'detection_confidence' not in face_annotations[0]:
                continue

            image_filename = json_filename[:-5] + '.jpg'
            image_file = os.path.join(image_dir, image_filename)
            image = Image.open(image_file)
            image_width, image_height = image.size
            bounding_boxes = []
            for face in face_annotations:
                bounding_boxes.append(face['bounding_poly']['vertices'])

            areas = []
            for box in bounding_boxes:
                width = box[1]['x'] - box[0]['x']
                height = box[2]['y'] - box[0]['y']
                area = width * height
                areas.append(area)
            largest_box_index = areas.index(max(areas))
            largest_bounding_box = bounding_boxes[largest_box_index]

            x = (largest_bounding_box[0]['x'] + largest_bounding_box[1]['x']) / 2
            y = (largest_bounding_box[0]['y'] + largest_bounding_box[2]['y']) / 2
            center_point = (x, y)

            point_x, point_y = center_point

            B = (image_width / 3, image_height / 3)
            C = (2 * image_width / 3, image_height / 3)
            D = (image_width / 3, 2 * image_height / 3)
            E = (2 * image_width / 3, 2 * image_height / 3)

            distance_AB = math.sqrt((B[0] - point_x) ** 2 + (B[1] - point_y) ** 2)
            distance_AC = math.sqrt((C[0] - point_x) ** 2 + (C[1] - point_y) ** 2)
            distance_AD = math.sqrt((D[0] - point_x) ** 2 + (D[1] - point_y) ** 2)
            distance_AE = math.sqrt((E[0] - point_x) ** 2 + (E[1] - point_y) ** 2)

            distance = min(distance_AB, distance_AC, distance_AD, distance_AE)
            writer.writerow([json_filename, image_filename, distance])


# 用于计算Shoe的bounding box
def BoundingBoxArea(bounding_box, image_width, image_height):
    width = bounding_box[2]["x"] - bounding_box[0]["x"]
    height = bounding_box[2]["y"] - bounding_box[0]["y"]

    width = width * image_width
    height = height * image_height

    area = width * height
    return area


def Shoe2Diagonal(Gjson_path, post_dir, output_dir):
    json_dir = Gjson_path
    image_dir = os.path.join(post_dir, os.path.basename(json_dir))
    folder_name = os.path.basename(os.path.normpath(Gjson_path))
    output_path = os.path.join(output_dir, f'{folder_name}_Shoe2Diag.csv')

    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['JSON File Name', 'Image File Name', 'Minimum Distance'])

        for json_filename in os.listdir(json_dir):
            if not json_filename.endswith('.json'):
                continue

            json_file = os.path.join(json_dir, json_filename)
            with open(json_file, 'r') as f:
                data = json.load(f)
                for annotation in data['object_annotations']:
                    if annotation['name'] == 'Shoe': 3

                    image_filename = json_filename[:-5] + '.jpg'
                    image_file = os.path.join(image_dir, image_filename)
                    image = Image.open(image_file)
                    image_width, image_height = image.size

                    max_bounding_box = None
                    max_center = None

                    for obj in data["object_annotations"]:
                        if obj["name"] == "Shoe":
                            bounding_box = obj["bounding_poly"]["normalized_vertices"]
                            area = BoundingBoxArea(bounding_box, image_width, image_height)
                            center_x = (bounding_box[0]["x"] + bounding_box[2]["x"]) / 2
                            center_y = (bounding_box[0]["y"] + bounding_box[2]["y"]) / 2
                            center = (center_x, center_y)
                            if max_bounding_box is None or area > BoundingBoxArea(max_bounding_box, image_width, image_height):
                                max_bounding_box = bounding_box
                                max_center = center
                    if max_center is None:
                        continue
                    point_x, point_y = max_center
                    point_x = point_x * image_width
                    point_y = point_y * image_height

                    k1 = image_height / image_width
                    b1 = 0
                    k2 = -k1
                    b2 = image_height

                    k_perp_1 = -1 / k1
                    b_perp_1 = point_y - point_x * k_perp_1

                    k_perp_2 = -1 / k2
                    b_perp_2 = point_y - point_x * k_perp_2

                    x_f1 = (b_perp_1 - b1) / (k1 - k_perp_1)
                    y_f1 = k1 * x_f1 + b1
                    x_f2 = (b_perp_2 - b2) / (k2 - k_perp_2)
                    y_f2 = k2 * x_f2 + b2
                    distance_ab = math.sqrt((x_f1 - point_x) ** 2 + (y_f1 - point_y) ** 2)
                    distance_ac = math.sqrt((x_f2 - point_x) ** 2 + (y_f2 - point_y) ** 2)
                    with open(output_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([json_filename, image_filename, min(distance_ab, distance_ac)])


def Shoe2IntersectionPoint(Gjson_path, post_dir, output_dir):
    json_dir = Gjson_path
    image_dir = os.path.join(post_dir, os.path.basename(json_dir))
    folder_name = os.path.basename(os.path.normpath(Gjson_path))
    output_path = os.path.join(output_dir, f'{folder_name}_Shoe2Point.csv')

    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['JSON File Name', 'Image File Name', 'Minimum Distance'])

        for json_filename in os.listdir(json_dir):
            if not json_filename.endswith('.json'):
                continue

            json_file = os.path.join(json_dir, json_filename)
            with open(json_file, 'r') as f:
                data = json.load(f)
                for annotation in data['object_annotations']:
                    if annotation['name'] == 'Shoe': 3

                    image_filename = json_filename[:-5] + '.jpg'
                    image_file = os.path.join(image_dir, image_filename)
                    image = Image.open(image_file)
                    image_width, image_height = image.size

                    max_bounding_box = None
                    max_center = None

                    for obj in data["object_annotations"]:
                        if obj["name"] == "Shoe":
                            bounding_box = obj["bounding_poly"]["normalized_vertices"]
                            area = BoundingBoxArea(bounding_box, image_width, image_height)
                            center_x = (bounding_box[0]["x"] + bounding_box[2]["x"]) / 2
                            center_y = (bounding_box[0]["y"] + bounding_box[2]["y"]) / 2
                            center = (center_x, center_y)
                            if max_bounding_box is None or area > BoundingBoxArea(max_bounding_box, image_width, image_height):
                                max_bounding_box = bounding_box
                                max_center = center
                    if max_center is None:
                        continue
                    point_x, point_y = max_center
                    point_x = point_x * image_width
                    point_y = point_y * image_height

                    B = (image_width / 3, image_height / 3)
                    C = (2 * image_width / 3, image_height / 3)
                    D = (image_width / 3, 2 * image_height / 3)
                    E = (2 * image_width / 3, 2 * image_height / 3)

                    distance_AB = math.sqrt((B[0] - point_x) ** 2 + (B[1] - point_y) ** 2)
                    distance_AC = math.sqrt((C[0] - point_x) ** 2 + (C[1] - point_y) ** 2)
                    distance_AD = math.sqrt((D[0] - point_x) ** 2 + (D[1] - point_y) ** 2)
                    distance_AE = math.sqrt((E[0] - point_x) ** 2 + (E[1] - point_y) ** 2)

                    with open(output_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [json_filename, image_filename, min(distance_AB, distance_AC, distance_AD, distance_AE)])



