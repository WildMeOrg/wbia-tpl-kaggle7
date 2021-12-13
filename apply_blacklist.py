# -*- coding: utf-8 -*-
with open('data/train.csv', 'r') as fr:
    lines = fr.readlines()
with open('data/blacklist_verified_and_halves.csv', 'r') as bf:
    black_lines = bf.readlines()
    bbb = set()
    for line in black_lines:
        bbb.add(line.strip())
with open('data/train_clean_no_halves.csv', 'w') as ff:
    for line in lines:
        if line.strip().split(',')[0] not in bbb:
            ff.write(line.strip() + '\n')
