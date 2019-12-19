from os.path import abspath, join, exists
import numpy as np
import utool as ut
import vtool as vt
import random
import uuid
import tqdm
import cv2


# ibs = None


MIN_AIDS = 1
MAX_AIDS = 20
MAX_NAMES = np.inf
PADDING = 32


url = 'https://cthulhu.dyn.wildme.io/public/random/humpback.all.csv'
local_filepath = ut.download_url(url)
filepath = abspath(local_filepath)

with open(filepath, 'r') as file:
    # header = file.readline()
    lines = file.readlines()
header = ['acmID', 'individualID']

line_list = []
for line in lines:
    line = line.strip()
    if len(line) == 0:
        continue
    line = line.split(',')
    line_dict = dict(zip(header, line))
    line_list.append(line_dict)

flukebook_image_acmid_list = ut.take_column(line_list, 'acmID')
flukebook_name_text_list = ut.take_column(line_list, 'individualID')
assert len(flukebook_name_text_list) == len(flukebook_image_acmid_list)

flukebook_image_uuid_list = list(map(uuid.UUID, flukebook_image_acmid_list))
gid_list = ibs.get_image_gids_from_uuid(flukebook_image_uuid_list)
nid_list = ibs.get_name_rowids_from_text(flukebook_name_text_list)
ibs.delete_empty_nids()
current_name_list = ibs.get_name_texts(ibs.get_valid_nids())
unknown_name_list = list(set(flukebook_name_text_list) - set(current_name_list))

print('FB Num records: %d' % (len(flukebook_name_text_list), ))
print('FB Num unique acmids: %d' % (len(set(flukebook_image_acmid_list)), ))
print('FB Num unique names: %d' % (len(set(flukebook_name_text_list)), ))
print('FB Num unique gids: %d' % (len(set(gid_list)), ))
print('FB Num unique nids: %d' % (len(set(nid_list)), ))
print('FB Num unknown acmids: %d' % (gid_list.count(None), ))
print('FB Num unknown names: %d' % (len(unknown_name_list), ))

flag_list = [gid is not None for gid in gid_list]
gid_list                  = ut.compress(gid_list, flag_list)
flukebook_image_uuid_list = ut.compress(flukebook_image_uuid_list, flag_list)
flukebook_name_text_list  = ut.compress(flukebook_name_text_list, flag_list)
assert None not in gid_list

print('DE-NONE-ING')
print('FB Num records: %d' % (len(flukebook_name_text_list), ))
print('FB Num unique acmids: %d' % (len(set(flukebook_image_acmid_list)), ))
print('FB Num unique names: %d' % (len(set(flukebook_name_text_list)), ))


# FIND DETECTIONS
depc = ibs.depc_annot

config = {'algo': 'lightnet', 'config_filepath' : 'candidacy', 'nms_thresh': 0.40, 'sensitivity': 0.4}
results_list = ibs.depc_image.get_property('localizations', gid_list, None, config=config)

global_gid_list  = []
global_bbox_list = []
global_name_list = []
for gid, result_list, flukebook_name_text in zip(gid_list, results_list, flukebook_name_text_list):
    status, bbox_list, theta_list, conf_list, species_list = result_list
    zipped = sorted(list(zip(conf_list, bbox_list, theta_list, species_list)), reverse=True)
    for conf, bbox, theta, species in zipped:
        if conf >= 0.90 and species == 'whale_fluke':
            global_gid_list.append(gid)
            global_bbox_list.append(bbox)
            global_name_list.append(flukebook_name_text)
            break

global_aid_list = ibs.add_annots(global_gid_list, global_bbox_list)

seen_aid_set = set([])
flag_list = []
for global_aid in global_aid_list:
    flag = global_aid not in seen_aid_set
    flag_list.append(flag)
    seen_aid_set.add(global_aid)

global_aid_list  = ut.compress(global_aid_list, flag_list)
global_name_list = ut.compress(global_name_list, flag_list)
ibs.set_annot_names(global_aid_list, global_name_list)

global_name_dict = {}
for global_aid, global_name in zip(global_aid_list, global_name_list):
    if global_name not in global_name_dict:
        global_name_dict[global_name] = set([])
    global_name_dict[global_name].add(global_aid)

aid_list = []
valid_aid_set = set([])
# test_aid_set = set([])
count_list = []
key_list = list(global_name_dict.keys())
random.shuffle(key_list)
for global_name in key_list:
    global_aid_set = global_name_dict[global_name]
    global_aid_list_ = list(global_aid_set)

    if len(global_aid_set) < MIN_AIDS:
        continue

    random.shuffle(global_aid_list_)

    if len(global_aid_list_) > MAX_AIDS:
        global_aid_list_ = global_aid_list_[:MAX_AIDS]

    if len(global_aid_list_) > 1:
        valid_aid = global_aid_list_[0]
        valid_aid_set.add(valid_aid)

    # test_aid = global_aid_list_[1]
    # test_aid_set.add(test_aid)

    aid_list += global_aid_list_
    count = len(global_aid_list_)
    count_list.append(count)
    if len(count_list) >= MAX_NAMES:
        break
print(ut.repr3(ut.dict_hist(count_list)))
print(ut.repr3(len(count_list)))

tips_list = depc.get('Notch_Tips', aid_list)
size_list = depc.get('chips', aid_list, ('width', 'height'))
config = {
    'dim_size': 1000,
    'resize_dim': 'width',
    'ext': '.jpg',
}
chip_list = depc.get('chips', aid_list, 'img', config=config, ensure=True)

color_list = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
]

tps = cv2.createThinPlateSplineShapeTransformer()

notch_path = 'notches'
ut.delete(notch_path)
ut.ensuredir(notch_path)

path_dict = {}
zipped = list(zip(aid_list, tips_list, size_list, chip_list))
for aid, tip_list, size, chip in tqdm.tqdm(zipped):
    h0, w0, c0 = chip.shape

    size = np.array(size, dtype=np.float32)

    notch = tip_list[0].copy()
    left  = tip_list[1].copy()
    right = tip_list[2].copy()

    notch /= size
    left  /= size
    right /= size

    size = np.array([w0, h0], dtype=np.float32)

    notch *= size
    left  *= size
    right *= size

    location_list = [
        tuple(map(int, np.around(notch))),
        tuple(map(int, np.around(left))),
        tuple(map(int, np.around(right))),
    ]
    chip_ = chip.copy()
    for location, color in zip(location_list, color_list):
        cv2.circle(chip_, location, 5, color=color)

    chip_filename = 'img_aid_%d_0.png' % (aid, )
    chip_filepath = join(notch_path, chip_filename)
    cv2.imwrite(chip_filepath, chip_)

    chip1 = chip.copy()
    h0, w0, c0 = chip1.shape

    left += PADDING
    notch += PADDING
    right += PADDING

    pad = np.zeros((h0, PADDING, 3), dtype=chip1.dtype)
    chip1 = np.hstack((pad, chip1, pad))
    h, w, c = chip1.shape
    pad = np.zeros((PADDING, w, 3), dtype=chip1.dtype)
    chip1 = np.vstack((pad, chip1, pad))
    h, w, c = chip1.shape

    location_list = [
        tuple(map(int, np.around(notch))),
        tuple(map(int, np.around(left))),
        tuple(map(int, np.around(right))),
    ]
    chip1_ = chip1.copy()
    for location, color in zip(location_list, color_list):
        cv2.circle(chip1_, location, 5, color=color)

    chip_filename = 'img_aid_%d_1.png' % (aid, )
    chip_filepath = join(notch_path, chip_filename)
    cv2.imwrite(chip_filepath, chip1_)

    delta = right - left
    radian = np.arctan2(delta[1], delta[0])
    degree = np.degrees(radian)
    M = cv2.getRotationMatrix2D((left[1], left[0]), degree, 1)
    chip2 = cv2.warpAffine(chip1, M, (w, h), flags=cv2.INTER_LANCZOS4)

    H = np.vstack((M, [0, 0, 1]))
    vert_list = np.array([notch, left, right])
    vert_list_ = vt.transform_points_with_homography(H, vert_list.T).T
    notch, left, right = vert_list_

    location_list = [
        tuple(map(int, np.around(notch))),
        tuple(map(int, np.around(left))),
        tuple(map(int, np.around(right))),
    ]
    chip2_ = chip2.copy()
    for location, color in zip(location_list, color_list):
        cv2.circle(chip2_, location, 5, color=color)

    chip_filename = 'img_aid_%d_2.png' % (aid, )
    chip_filepath = join(notch_path, chip_filename)
    cv2.imwrite(chip_filepath, chip2_)

    tps.clear()

    left[0]  -= PADDING // 2
    left[1]  -= PADDING // 2
    notch[1] += PADDING // 2
    right[0] += PADDING // 2
    right[1] -= PADDING // 2

    sshape = np.array([left, notch, right], np.float32)
    tshape = np.array([[0, 0], [w0 // 2, h0], [w0, 0]], np.float32)
    sshape = sshape.reshape(1, -1, 2)
    tshape = tshape.reshape(1, -1, 2)
    matches = [
        cv2.DMatch(0, 0, 0),
        cv2.DMatch(1, 1, 0),
        cv2.DMatch(2, 2, 0),
    ]
    tps.estimateTransformation(tshape, sshape, matches)
    chip3 = tps.warpImage(chip2)

    chip_filename = 'img_aid_%d_3.png' % (aid, )
    chip_filepath = join(notch_path, chip_filename)
    cv2.imwrite(chip_filepath, chip3)

    chip4 = chip3[:h0, :w0, :]

    chip_filename = 'img_aid_%d.png' % (aid, )
    chip_filepath = join(notch_path, chip_filename)
    cv2.imwrite(chip_filepath, chip4)
    path_dict[aid] = abspath(chip_filepath)

##########

output_path = join(ibs.dbdir, 'export-encounters')
output_path_train = join(output_path, 'train')
output_path_valid = join(output_path, 'valid')
# output_path_test = join(output_path, 'test')
output_path_train_folders = join(output_path_train, 'folders')
output_path_valid_folders = join(output_path_valid, 'folders')
# output_path_test_folders = join(output_path_test, 'folders')
output_path_train_manifest = join(output_path_train, 'manifest')
output_path_valid_manifest = join(output_path_valid, 'manifest')
# output_path_test_manifest = join(output_path_test, 'manifest')

assert not exists(output_path)
ut.delete(output_path)
ut.ensuredir(output_path)
ut.ensuredir(output_path_train)
ut.ensuredir(output_path_valid)
# ut.ensuredir(output_path_test)
ut.ensuredir(output_path_train_folders)
ut.ensuredir(output_path_valid_folders)
# ut.ensuredir(output_path_test_folders)
ut.ensuredir(output_path_train_manifest)
ut.ensuredir(output_path_valid_manifest)
# ut.ensuredir(output_path_test_manifest)

# gid_list = ibs.get_annot_gids(aid_list)
# uuid_list = ibs.get_image_uuids(gid_list)
uuid_list = ibs.get_annot_visual_uuids(aid_list)
uuid_str_list = list(map(str, uuid_list))
chip_filepath_list = ut.take(path_dict, aid_list)
name_text_list = ibs.get_annot_names(aid_list)

manifest_dict = {
    'train': [],
    'valid': [],
    # 'test': [],
}
zipped = sorted(list(zip(aid_list, uuid_str_list, chip_filepath_list, name_text_list)))
for aid, uuid_str, chip_filepath, name_text in tqdm.tqdm(zipped):
    chip_filepath = abspath(chip_filepath)
    assert exists(chip_filepath)

    named_humpback_unixtime = ibs.get_annot_image_unixtimes(aid)

    version = 'train'
    output_path_folders = output_path_train_folders
    output_path_manifest = output_path_train_manifest

    if aid in valid_aid_set:
        # assert aid not in test_aid_set
        version = 'valid'
        output_path_folders = output_path_valid_folders
        output_path_manifest = output_path_valid_manifest

    # if aid in test_aid_set:
    #     assert aid not in valid_aid_set
    #     version = 'test'
    #     output_path_folders = output_path_test_folders
    #     output_path_manifest = output_path_test_manifest

    name_output_path = join(output_path_folders, name_text)
    if not exists(name_output_path):
        ut.ensuredir(name_output_path)
    annot_output_filepath = join(name_output_path, '%s.jpg' % (uuid_str, ))
    assert not exists(annot_output_filepath)
    ut.copy(chip_filepath, annot_output_filepath, verbose=False)

    annot_output_filename = '%s.jpg' % (uuid_str, )
    annot_output_filepath = join(output_path_manifest, annot_output_filename)
    assert not exists(annot_output_filepath)
    ut.copy(chip_filepath, annot_output_filepath, verbose=False)
    manifest_line = '%s,%s,%s' % (annot_output_filename, name_text, named_humpback_unixtime, )
    manifest_dict[version].append(manifest_line)

for manifest_key in manifest_dict:
    manifest_list = manifest_dict[manifest_key]
    manifest_list = sorted(manifest_list)
    manifest_list = ['Image,ID,Unixtime'] + manifest_list
    manifest_str = '\n'.join(manifest_list)
    manifest_filepath = join(output_path, '%s.txt' % (manifest_key, ))
    with open(manifest_filepath, 'w') as manifest_file:
        manifest_file.write(manifest_str)
