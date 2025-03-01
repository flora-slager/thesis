#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from datetime import datetime
import glob
import json
import os
import re
import sys
import tensorflow as tf

# Allow relative imports when being executed as script.
from natsort import natsort

from bin.evaluate_ensemble import evaluate_dual_memory_model

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401

    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# sess = tf.compat.v1.Session(config=config)

def create_generator(args, preprocess_image):
    """ Create generators for evaluation.
    """
    common_args = {
        'preprocess_image': preprocess_image,
    }

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from keras_retinanet.preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'pascal':
        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            image_extension=args.image_extension,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--image-extension', help='Declares the dataset images\' extension.', default='.jpg')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',
                        help='Convert the model to an inference model (ie. the input is a training model).',
                        action='store_true')
    parser.add_argument('--backbone', help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).', type=int)
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).',
                        default=0.05, type=float)
    parser.add_argument('--iou-threshold', help='IoU Threshold to count for a positive detection (defaults to 0.5).',
                        default=0.5, type=float)
    parser.add_argument('--max-detections', help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path', help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.',
                        type=int, default=1333)
    parser.add_argument('--config',
                        help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)

# From training pipeline, find and load most recent historical model:

def GetSubfolders(folder):
    return [folder for folder in glob.glob(os.path.join(folder, "*"))
            if os.path.isdir(folder)]


def LoadMostRecentModel(historical_snapshots_folder):
    print("Looking in", os.path.abspath(historical_snapshots_folder))

    subfolders = GetSubfolders(historical_snapshots_folder)
    if len(subfolders) == 0:
        return None
    most_recent_subfolder = natsort.natsorted(subfolders)[-1]
    filenames = glob.glob(os.path.join(most_recent_subfolder, "*"))
    most_recent_filename = natsort.natsorted(filenames)[-1]
    return most_recent_filename


def LoadModels(historical_snapshots_folder, backbone, day_number=None):
    print("Started dual-memory modelling, looking in", historical_snapshots_folder)

    # If there is no "most_recent_snapshot", return None:

    most_recent_snapshot = LoadMostRecentModel(historical_snapshots_folder)
    if most_recent_snapshot is None:
        return None

    print("Most recent snapshot:", most_recent_snapshot)

    # Search folder for a Day-10 snapshot:
    # f"historical_snapshots/Day{day_number}/snapshots/"

    if day_number is None:
        match = re.search("Day(\d+)", most_recent_snapshot)
        if match is None:
            raise ValueError("Filename doesn't conform to standard")

        day_number = int(match.group(1))
    print("Day number is:", day_number)

    if day_number > 10:
        print("Day number is greater than 10")
        find_day = day_number - 10
        folder = os.path.join(historical_snapshots_folder, f"Day{find_day}/")
        filenames = glob.glob(os.path.join(folder, "*"))
        combine_model_filename = natsort.natsorted(filenames)[-1]
        print("Done")

        # load and combine models:
        # models.load_model(model_filename, backbone_name=args.backbone)
        all_models = [models.load_model(most_recent_snapshot, backbone_name=backbone),
                      models.load_model(combine_model_filename, backbone_name=backbone)]

        return all_models
    else:
        return [models.load_model(most_recent_snapshot, backbone_name=backbone)]


def main(args=None, model_filename=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

    # make sure keras and tensorflow are the minimum required version
    check_keras_version()
    check_tf_version()

    # optionally choose specific GPU
    if args.gpu:
        setup_gpu(args.gpu)

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    backbone = models.backbone(args.backbone)
    generator = create_generator(args, backbone.preprocess_image)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model, this may take a second...')
    if args.continual_learning_model == 'dual_memory': # Continual learning dual-memory modelling treatment
        base_models = LoadModels(args.historical_snapshots_folder, args.backbone, args.day_number)
        all_models = []
        for model in base_models:
            generator.compute_shapes = make_shapes_callback(model)
            if args.convert_model:
                model = models.convert_model(model, anchor_params=anchor_params)
            all_models.append(model)

        (average_precisions, inference_time,
         detections_per_model, final_detections) = evaluate_dual_memory_model(
            generator,
            all_models,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            save_path=args.save_path
        )

        # bbox_savepath given, save bounding box coordinates from dual-memory model predictions:

        if args.bbox_savepath:
            detections_per_model = [[[class_predictions.tolist() for class_predictions in image]
                                    for image in model_predictions]
                                    for model_predictions in detections_per_model]
            detections_with_filenames = {'final_detections': final_detections, 'annotations': args.annotations,
                                        'detections_per_model': detections_per_model}
            with open(args.bbox_savepath, 'wt') as outf:
                json.dump(detections_with_filenames, outf)

            print("Finished dual memory model")
            print(average_precisions, inference_time)

    else:
        if model_filename is None:
            model_filename = args.model
        model = models.load_model(model_filename, backbone_name=args.backbone)

        generator.compute_shapes = make_shapes_callback(model)

        # optionally convert the model
        if args.convert_model:
            model = models.convert_model(model, anchor_params=anchor_params)

        # print model summary
        # print(model.summary())

        # start evaluation
        if args.dataset_type == 'coco':
            from ..utils.coco_eval import evaluate_coco
            evaluate_coco(generator, model, args.score_threshold)
        else:
            average_precisions, inference_time = evaluate(
                generator,
                model,
                iou_threshold=args.iou_threshold,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                save_path=args.save_path
            )

    # print evaluation
    total_instances = []
    precisions = []
    #labels = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations),
                generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        #labels.append(label)
        total_instances.append(num_annotations)
        precisions.append(average_precision)

    if sum(total_instances) == 0:
        print('No test instances found.')
        return

    print('Inference time for {:.0f} images: {:.4f}'.format(generator.size(), inference_time))

    print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
        sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
    print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))

    #print(labels)
    print(precisions)
    print(total_instances)

    # Save mAP and other accuracy statistics to mAP_savepath:
    
    mAP = sum(precisions) / sum(x > 0 for x in total_instances)
    date = datetime.now().strftime("%Y%m%d%H%M")
    with open(args.mAP_savepath, 'a') as outf:
        outf.write(f"{date}, {mAP}, {precisions}, {total_instances}, {model_filename}, {args.continual_learning_model}" + "\n")
    return mAP

if __name__ == '__main__':
    main()
