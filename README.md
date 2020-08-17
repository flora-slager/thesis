# Erasmus School of Economics MSc Thesis Data Science and Marketing Analytics 

### A step-wise guide and source code with which to repeat the methodology conducted in this thesis.

#### Organise `keras_retinanet` directory.
1. Download the `fizyr/keras-retinanet` repository from https://github.com/fizyr/keras-retinanet as `keras_retinanet`.
2. Within `keras_retinanet/keras_retinanet` save `training_pipeline.py`.
3. Within `keras_retinanet/keras_retinanet/bin`, delete `train.py` and `evaluate.py`, and replace with the `train.py` and `evaluate.py` files from this repository. 
4. Add `evaluate_ensemble.py` to `keras_retinanet/keras_retinanet/bin`.
5. Within `keras_retinanet/keras_retinanet`, create the folder `data`. Within this folder, create the following empty folders: `historical`; `train`; and `val`.
6. Download the `resnet50_coco_best_v2.1.0.h5` from https://github.com/fizyr/keras-retinanet/releases and save to `data`.
7. Save `class_map.csv` to `data`.

#### Organise `coco_data` directory.
1. Outside of `keras_retinanet` create `coco_data` folder.
2. Create the following empty folders: `images` and `annotations`.
3. Create the following empty folders within `images`: `train_subset` and `val_subset`.
4. From https://cocodataset.org/#download download `2017 Train images`, `2017 Val images` as `train2017`, `val2017` into `images`.
5. From https://cocodataset.org/#download download `2017 Train/Val annotations` into `annotations`. 
6. Save `coco_subset_generator.py` to `coco_data`.

#### Generate training and validation image subfolders.
1. To generate the training subset, run `coco_subset_generator.py` through command line as 
```
coco_subset_generator.py --annotations=annotations/instances_train2017.json --source=images/train2017/ --subset=images/train_subset/
```
2. To generate the validation subset, run `coco_subset_generator.py` through command line as 
```
coco_subset_generator.py --annotations=annotations/instances_val2017.json --source=images/val2017/ --subset=images/val_subset/
```

#### Generate training and validation CSV files.
1. To generate the training CSV files from Experiment A and B, run `coco_csv_generator.py` through command line as
```
coco_csv_generator.py --





