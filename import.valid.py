from os.path import join, splitext
import wbia
import uuid

ibs = wbia.opendb(dbdir='/data/wbia/testdb_kaggle7/')
ibs_src = wbia.opendb(dbdir='/data/wbia/Flukebook_Master/')

data_filepath = '/home/jason.parham/'
with open(join(data_filepath, 'valid.txt')) as valid_file:
    header = valid_file.readline()
    valid_line_list = valid_file.readlines()

annot_uuid_str_list = []
name_text_list = []

header = header.strip().split(',')
for valid_line in valid_line_list:
    valid_line = valid_line.strip().split(',')
    image_filename, name_text, unixtime = valid_line
    annot_uuid_str, ext = splitext(image_filename)

    annot_uuid_str_list.append(annot_uuid_str)
    name_text_list.append(name_text)

annot_visual_uuid_list = list(map(uuid.UUID, annot_uuid_str_list))
src_aid_list = ibs_src.get_annot_aids_from_visual_uuid(annot_visual_uuid_list)
src_gid_list = ibs_src.get_annot_gids(src_aid_list)

assert None not in src_aid_list
assert None not in src_gid_list

gpath_list = ibs_src.get_image_paths(src_gid_list)
gid_list = ibs.add_images(gpath_list)

bbox_list = ibs_src.get_annot_bboxes(src_aid_list)
name_list = ibs_src.get_annot_names(src_aid_list)

for name1, name2 in zip(name_text_list, name_list):
    assert name1 == name2

aid_list = ibs.add_annots(gid_list, bbox_list)
ibs.set_annot_names(aid_list, name_list)
