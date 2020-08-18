# Erasmus School of Economics MSc Thesis Data Science and Marketing Analytics 

### A step-wise guide and source code with which to repeat the methodology conducted in this thesis.

#### Organise `keras_retinanet` directory.
1. Download the `fizyr/keras-retinanet` repository from https://github.com/fizyr/keras-retinanet as `keras_retinanet`.
2. Within `keras_retinanet/keras_retinanet` save `training_pipeline.py`, which is provided in this repository.
3. Within `keras_retinanet/keras_retinanet/bin`, delete `train.py` and `evaluate.py`, and replace with the `train.py` and `evaluate.py` files from this repository. 
4. Add `evaluate_ensemble.py` to `keras_retinanet/keras_retinanet/bin`.
5. Within `keras_retinanet/keras_retinanet`, create the folders `data` and `results`. 
6. Within the `data` folder, create the following empty folders: `historical`; `train`; and `val`.
7. Download the `resnet50_coco_best_v2.1.0.h5` weights file from https://github.com/fizyr/keras-retinanet/releases and save it to `data`.
8. Save `class_map.csv` to `data`.

#### Organise `coco_data` directory.
1. Outside of `keras_retinanet` create a `coco_data` folder.
2. Create the following empty folders: `images` and `annotations`.
3. Create the following empty folders within `images`: `train_subset` and `val_subset`.
4. From https://cocodataset.org/#download download `2017 Train images`, `2017 Val images` as `train2017`, `val2017` into `images`.
5. From https://cocodataset.org/#download download `2017 Train/Val annotations` into `annotations`. 
6. Save `coco_subset_generator.py` to `coco_data`.
7. Save `coco_csv_generator.py` to `coco_data`.

#### Generate training and validation image subfolders.
1. Generate the training subset by running `coco_subset_generator.py` through command line as 
```
coco_subset_generator.py --annotations=annotations/instances_train2017.json --source=images/train2017/ --subset=images/train_subset/
```
2. Generate the validation subset by running `coco_subset_generator.py` through command line as 
```
coco_subset_generator.py --annotations=annotations/instances_val2017.json --source=images/val2017/ --subset=images/val_subset/
```

#### Run tuning benchmark tests.
1. Generate the training CSV files for tuning by running `coco_csv_generator.py` through command line as
```
coco_csv_generator.py --path_to_images=images/train_subset/ --num_files=600 --path_to_json=annotations/instances_train2017.json --csv_save_path=keras_retinanet/keras_retinanet/data/train/ --files=7 --tuning:person
```
2. Generate the validation CSV files for tuning by running `coco_csv_generator.py` through command line as
```
coco_csv_generator.py --path_to_images=images/train_subset/ --num_files=150 --path_to_json=annotations/instances_train2017.json --csv_save_path=keras_retinanet/keras_retinanet/data/train/ --files=7 --tuning:person
```
3. Alter `class_map.csv` by removing all but the first row.
4. Set up batch size benchmark test by altering `train_config.json` and `val_config.json` as follows: `gpu:0`; `batch_size:1`; `epochs:10`; `steps:600`; `continual_learning_model:null`; and `regularisation:3.0`, for batch size 1, and altering `batch_size:8` for batch size 8. The savepath arguments can remain default or altered to any name deemed informative.
5. Conduct batch size benchmark test by running `training_pipeline.py` for batch size 1 and batch size 8 through command line as
```
training_pipeline.py --num_repeat=7
```
6. Clear `data/historical`, and save the mAP, losses and time results to `results`.
7. Set up epoch benchmark test by altering `epochs:600` in `train_config.json` and `val_config.json`. The savepath arguments can remain default or altered to any name deemed informative.
8. Conduct epoch benchmark test by running `training_pipeline.py` through command line as
```
training_pipeline.py --num_repeat=1
```
9. Clear `data/historical`, and save the mAP, losses and time results to `results`.
10. Continue epoch benchmark test by altering `train_config.json` and `val_config.json` to `epochs:5`, `epochs:10`, and `epochs:25`. The savepath arguments can remain default or altered to any name deemed informative.
11. Conduct continued epoch benchmark test by running `training_pipeline.py` for each alteration of epochs through command line as
```
training_pipeline.py --num_repeat=7
```
12. Clear `data/historical`, and save the mAP, losses and time results to `results`.

#### Conduct Experiment A.
1. Generate the training CSV files for Experiment A by running `coco_csv_generator.py` through command line as
```
coco_csv_generator.py --path_to_images=images/train_subset/ --num_files=600 --path_to_json=annotations/instances_train2017.json --csv_save_path=keras_retinanet/keras_retinanet/data/train/ --files=90
```
2. Generate the validation CSV files for Experiment A by running `coco_csv_generator.py` through command line as
```
coco_csv_generator.py --path_to_images=images/val_subset/ --num_files=150 --path_to_json=annotations/instances_val2017.json --csv_save_path=keras_retinanet/keras_retinanet/data/val/ --files=90
```
3. Return `class_map.csv` to original state.
4. Set up `train_config.json` and `val_config.json` for use in the Experiment A Baseline training simulation by setting `gpu:0`, `batch_size:1`, `epochs:10`, `steps:600`, `continual_learning_model:null`, and `regularisation:3.0`. The savepath arguments can remain default or altered to any name deemed informative.
5. Conduct the Baseline training simulation for Experiment A by running `training_pipeline.py` through command line as
```
training_pipeline.py --num_repeat=90
```
6. Clear `data/historical`, and save the mAP, losses and time results to `results`.
7. Set up `train_config.json` and `val_config.json` for use in the Experiment A regularisation training simulation by altering `regularisation:1.0`. The savepath arguments can remain default or altered to any name deemed informative.
8. Conduct the regularisation training simulation for Experiment A by running `training_pipeline.py` through command line as
```
training_pipeline.py --num_repeat=90
```
9. Clear `data/historical`, and save the mAP, losses and time results to `results`.
10. Set up `train_config.json` and `val_config.json` for use in the Experiment A regularisation training simulation by altering `regularisation:3.0` and `continual_learning_model:dual_memory`. The savepath arguments can remain default or altered to any name deemed informative.
11. Conduct the regularisation training simulation for Experiment A by running `training_pipeline.py` through command line as
```
training_pipeline.py --num_repeat=90
```
12. Clear `data/historical`, and save the mAP, losses and time results to `results`.

#### Conduct Experiment B.
1. Repeat the steps outlined for Experiment A.

#### Conduct Experiment C.
1. Generate the training CSV files for Experiment C by running `coco_csv_generator.py` through command line as
```
coco_csv_generator.py --path_to_images=images/train_subset/ --num_files=9000 --path_to_json=annotations/instances_train2017.json --csv_save_path=keras_retinanet/keras_retinanet/data/train/ --files=7
```
2. Generate the validation CSV files for Experiment C by running `coco_csv_generator.py` through command line as
```
coco_csv_generator.py --path_to_images=images/val_subset/ --num_files=384 --path_to_json=annotations/instances_val2017.json --csv_save_path=keras_retinanet/keras_retinanet/data/val/ --files=7
```
3. Repeat remaining steps outlined for Experiment A.
