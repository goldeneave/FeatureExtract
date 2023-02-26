# the first version of face2diag
def Face2Diagonal(Gjson_path, post_dir, output_dir):
    json_dir = Gjson_path
    image_dir = os.path.join(post_dir, os.path.basename(json_dir))
    folder_name = os.path.basename(os.path.normpath(Gjson_path))
    output_path = os.path.join(output_dir, f'{folder_name}_Face2Diag.csv')

    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['JSON File Name', 'Image File Name', 'Minimum Distance'])

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

            with open(output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([json_filename, image_filename, min(distance_ab, distance_ac)])


# first version of face2point
def Face2IntersectionPoint(Gjson_path, post_dir, output_dir):
    json_dir = Gjson_path
    image_dir = os.path.join(post_dir, os.path.basename(json_dir))
    folder_name = os.path.basename(os.path.normpath(Gjson_path))
    output_path = os.path.join(output_dir, f'{folder_name}_Face2Point.csv')

    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['JSON File Name', 'Image File Name', 'Minimum Distance'])

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

            with open(output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([json_filename, image_filename, min(distance_AB, distance_AC, distance_AD, distance_AE)])

