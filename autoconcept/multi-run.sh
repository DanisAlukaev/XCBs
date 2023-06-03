#!/bin/bash
for i in {49..62}
do
    for j in 42 0 17 9 3
    do
        python main.py -m dataset.batch_size=64 seed=$j +experiment=E$i-SHP
    done
done
