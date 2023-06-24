#!/bin/bash

# SHP pre-trained
# for i in {35..48}
# do
#     for j in 42 0 17 9 3
#     do
#         python main.py -m dataset.batch_size=64 seed=$j +experiment=E$i-SHP
#     done
# done

# SHP from scratch
# for i in {49..62}
# do
#     for j in 42 0 17 9 3
#     do
#         python main.py -m dataset.batch_size=64 seed=$j +experiment=E$i-SHP
#     done
# done

# MIM
# for i in 36 39
# do
#     for j in 42 0 17 9 3
#     do
#         python main.py -m dataset.batch_size=64 seed=$j +experiment=E$i-MIM
#     done
# done

# SHP-CBM
# for j in 42 0 17 9 3
# do
#     python main.py -m dataset.batch_size=64 seed=$j +experiment=E63-SHP
# done

# MIM-CBM
# for j in 42 0 17 9 3
# do
#     python main.py -m dataset.batch_size=64 seed=$j +experiment=E63-MIM
# done

# CUB
for j in 42 0 17 9 3
do
    python main.py -m dataset.batch_size=64 seed=$j +experiment=E39-CUB
done
