import os
import json
import re
import shutil
import itertools

import cv2
import numpy as np

DATASET_TYPE = 'preprocessed_merged' # small, medium, large etc
base_path = '/storage2/david/datasets/xview/raw/tier3_train_merge'
test_train_split = 10

images_dir = os.path.join(base_path, 'images')
labels_dir = os.path.join(base_path, 'labels')

# image_files = {}

def extract_filename(name):
    return name.split('_')

dataset = {}
for img_filename in os.listdir(images_dir):
    # extracting dataset information
    disaster_name, t, pre_or_post, _ = img_filename.split('_')

    if disaster_name not in dataset.keys():
        dataset[disaster_name] = set()

    dataset[disaster_name].add(int(t))

def parse_polygon(input_str):
    # Remove the "POLYGON (( * ))" with regex
    matches = re.search(r'\(\(([-\d. ,e]+)\)\)', input_str)
    assert matches, 'Failed to parse Polygon' + input_str
    coords = matches.group(1)
    ndarray = np.array([[comp for comp in coord.strip().split(' ')] for coord in coords.split(',')], dtype=np.float64)
    return ndarray


def load_json(disaster_name, idx, pre_or_post):
    file_name = f'{disaster_name}_{idx:08d}_{pre_or_post}_disaster.json'
    file_path = os.path.join(labels_dir, file_name)
    with open(file_path) as json_file:
        data = json.load(json_file)

    # os.path.join lng_lat with xy
    data_features = data['features']
    arr = []
    for xy, ll in zip(data_features['xy'], data_features['lng_lat']):
        obj = {}
        obj['properties'] = ll['properties']
        obj['xy'] = parse_polygon(xy['wkt'])
        obj['lng_lat'] = parse_polygon(ll['wkt'])
        arr.append(obj)
    data['features'] = arr
    return data


def get_damage_level(subtype):
    if subtype == 'no-damage':
        return 0
    elif subtype == 'minor-damage':
        return 1
    elif subtype == 'major-damage':
        return 2
    elif subtype == 'destroyed':
        return 3
    elif subtype == 'un-classified':
        return 4


def get_counterpart_filename(filename):
    filenames = filename.split('_')
    pre_or_post = filenames[-2]

    filenames[-2] = 'pre' if pre_or_post == 'post' else 'post'
    counterpart_filename = '_'.join(filenames)
    return counterpart_filename


def copy_to_dest(filename, dest_path):
    src_img = os.path.join(base_path, 'images', filename)
    tgt_img = os.path.join(dest_path, filename)
    shutil.copyfile(src_img, tgt_img)


def postprocessed_and_save(dataset_path, filename, mask):
    file_path = os.path.join(dataset_path, 'label_mask', filename)
    # print(file_path)
    cv2.imwrite(file_path, mask)


def xview2_to_coco(disaster_name, img_id, data_label, img_global_id):
    record = {}
    record['file_name'] = data_label['metadata']['img_name']
    record["height"] = 1024
    record['width'] = 1024
    record['disaster_name'] = disaster_name
    record['img_id'] = img_global_id
    objs = []
    # Add annotations (masks, bboxes etc)
    image_polygons_cv2 = []
    for feature in data_label['features']:
        polygon_xy = feature['xy']
        poly = list(itertools.chain.from_iterable(polygon_xy))
        px, py = polygon_xy.T
        if 'subtype' in feature['properties'].keys():
            damage_level = get_damage_level(feature['properties']['subtype'])
        else:
            damage_level = -1
        obj = {
            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            "bbox_mode": 0,  # BoxMode.XYXY_ABS
            "segmentation": [poly],
            "category_id": 0,
            "iscrowd": 0,
            'damage_level': damage_level,
        }
        objs.append(obj)
        image_polygons_cv2.append(np.array(polygon_xy, dtype=np.int32))

    # generate a rasterized image =====
    rasterized_image = np.zeros((1024, 1024), dtype=np.uint8)
    cv2.fillPoly(rasterized_image, image_polygons_cv2, 1)

    image_weight = int(rasterized_image.sum())
    record['image_weight'] = image_weight
    record['annotations'] = objs
    return record, rasterized_image


def CLAHE(image, clip_limit=3):
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert iamge from LAB color model back to RGB color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image


def VARI(image):
    # Input is in BGR
    B = image[..., 0]
    G = image[..., 1]
    R = image[..., 2]
    eps = 1e-6
    VARI = (G - R) / 2 * (eps + G + R - B) + 128  # Linearly transformed to be [0, 255]
    return VARI


output_base_path = f"/storage2/david/datasets/xview/{DATASET_TYPE}"
print('output path base', output_base_path)
train_path = os.path.join(output_base_path, 'train')
test_path = os.path.join(output_base_path, 'test')
train_mask_path = os.path.join(train_path, 'label_mask')
test_mask_path = os.path.join(test_path, 'label_mask')

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
os.makedirs(train_mask_path, exist_ok=True)
os.makedirs(test_mask_path, exist_ok=True)

train_label_file = os.path.join(train_path, 'labels.json')
test_label_file = os.path.join(test_path, 'labels.json')

test_set_locations = []

dataset_dicts_train = []
dataset_dicts_test = []

skipped_disasters = []  # 'palu-tsunami', 'sunda-tsunami'
img_global_id = 0
for idx, (disaster_name, image_ids) in enumerate(dataset.items()):

    # Skipping palu
    if disaster_name in skipped_disasters:
        print('skipping', disaster_name)
        continue

    max_polygon_in_disaster = 0
    mean_polygon_in_disaster = 0
    large_image_count = 0

    # Iterate through images
    for i, img_id in enumerate(image_ids):
        record = {
            'pre': {},
            'post': {},
        }
        img_global_id += 1
        data_label = load_json(disaster_name, img_id, 'post')
        record['post'], post_ras_img = xview2_to_coco(disaster_name, img_id, data_label, img_global_id)
        data_label = load_json(disaster_name, img_id, 'pre')
        record['pre'], pre_ras_img = xview2_to_coco(disaster_name, img_id, data_label, img_global_id)
        # end of rasterized image_generation

        mean_polygon_in_disaster += len(data_label['features'])
        if len(data_label['features']) > max_polygon_in_disaster:
            max_polygon_in_disaster = len(data_label['features'])

        # Splitting test with training set
        if disaster_name in test_set_locations:
            # TEST SET
            pre_filename = record['pre']['file_name']
            copy_to_dest(pre_filename, test_path)
            postprocessed_and_save(test_path, pre_filename, pre_ras_img)

            post_filename = get_counterpart_filename(pre_filename)
            copy_to_dest(post_filename, test_path)
            postprocessed_and_save(test_path, post_filename, post_ras_img)

            dataset_dicts_test.append(record)

        else:
            # TRAIN SET
            pre_filename = record['pre']['file_name']
            copy_to_dest(pre_filename, train_path)
            postprocessed_and_save(train_path, pre_filename, pre_ras_img)

            post_filename = get_counterpart_filename(pre_filename)
            copy_to_dest(post_filename, train_path)
            postprocessed_and_save(train_path, post_filename, post_ras_img)

            dataset_dicts_train.append(record)
        # generate debug set

    mean_polygon_in_disaster /= len(image_ids)
    print(train_path)
    print(
        f'{disaster_name}({i}),\t max_poly: {max_polygon_in_disaster},\t mean_poly: {int(mean_polygon_in_disaster)}, \t large_image_count: {large_image_count}')

with open(train_label_file, 'w') as outfile:
    json.dump(dataset_dicts_train, outfile)
with open(test_label_file, 'w') as outfile:
    json.dump(dataset_dicts_test, outfile)
print('saving to json complete!')