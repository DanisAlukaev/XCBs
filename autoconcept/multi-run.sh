#!/bin/bash

# SHP pre-trained
for i in {35..48}
do
    for j in 42 0 17 9 3
    do
        python main.py -m dataset.batch_size=64 seed=$j +experiment=E$i-SHP
    done
done

# SHP from scratch
# for i in {49..62}
# do
#     for j in 42 0 17 9 3
#     do
#         python main.py -m dataset.batch_size=64 seed=$j +experiment=E$i-SHP
#     done
# done
