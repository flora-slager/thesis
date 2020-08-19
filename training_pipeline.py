# %%
# Import packages:

import argparse
from datetime import datetime
from glob import glob
import json
import keras
import keras.preprocessing.image
import matplotlib as plt
import natsort
import os
import pandas as pd
import re
import sys
import tensorflow as tf
from timeit import default_timer as timer
import warnings
import winsound

from keras_retinanet import *
from keras_retinanet import models
from bin import evaluate
from bin import train

# %%
train_config_filename = r"train_config.json"
val_config_filename = r"val_config.json"

# %%

parser = argparse.ArgumentParser(description = "Training script for running the training pipeline.")
parser.add_argument("--num_repeat", help = "Number of times to repeat RunPipeline.") 
#parser.add_argument('--train_config', help = 'Training config json filepath.')   
#parser.add_argument('--val_config', help = 'Val config json filepath.')  
    
args = parser.parse_args()



# %%
# Create a folder for each day and move last snapshot:

def MoveSnapshots(snapshots_folder, day_number, historical_snapshots_folder):
    dest_folder = f"{historical_snapshots_folder}/Day{day_number}/"
    try:
        os.makedirs(dest_folder)
    except FileExistsError:
        pass
    filenames = os.listdir(snapshots_folder)
    print("\n" + "Files in", snapshots_folder, "are", filenames)
    last = natsort.natsorted(filenames)[-1]
    print("Latest snapshot is " + last)
    scr = os.path.join(snapshots_folder, last)
    print("Moving " + last + " to " + dest_folder)
    dest = os.path.join(dest_folder, last)
    print("New destination is " + dest)
    os.rename(scr, dest)

# %%
# Delete unneccesary snapshot files for safety:

def DeleteSnapshots(snapshots_folder):
    snapshots = os.listdir(snapshots_folder)
    print("\n" + "Snapshots to remove are", snapshots)
    for snapshot in snapshots:
        os.remove(os.path.join(snapshots_folder, snapshot))
    print("Deleted", snapshots)

# %%
# Gets day subfolder from historical snapshots folder:

def GetSubfolders(folder):
    return [folder for folder in glob(os.path.join(folder, "*"))
            if os.path.isdir(folder)]

# %%
# Loads most recent snapshot from most recent day:

def LoadMostRecentModel(historical_snapshots_folder):
    print("\n" + "Looking in", os.path.abspath(historical_snapshots_folder))

    subfolders = GetSubfolders(historical_snapshots_folder)
    if len(subfolders) == 0:
        return None
    most_recent_subfolder = natsort.natsorted(subfolders)[-1]
    filenames = glob(os.path.join(most_recent_subfolder, "*"))
    most_recent_filename = natsort.natsorted(filenames)[-1]
    return most_recent_filename
    print("\n" + "Most recent snapshot is", most_recent_filename)

# %%
# Load models for dual-memory modelling:

def LoadModels(historical_snapshots_folder, backbone):
    print("\n" + "Started dual-memory modelling, looking in", historical_snapshots_folder)

    # If there is no "most_recent_snapshot", return None:

    most_recent_snapshot = LoadMostRecentModel(historical_snapshots_folder)
    if most_recent_snapshot is None:
        return None

    print("Most recent snapshot:", most_recent_snapshot)

    # Search folder for a Day-10 snapshot:
    # f"historical_snapshots/Day{day_number}/snapshots/"

    match = re.search("Day(\d+)", most_recent_snapshot)
    if match is None:
        raise ValueError("Filename doesn't conform to standard")

    day_number = int(match.group(1))
    print("Day number is:", day_number)

    if day_number > 10:
        print("Day number is greater than 10")
        find_day = 0
        folder = os.path.join(historical_snapshots_folder, f"Day{find_day}/")
        filenames = glob(os.path.join(folder, "*"))
        combine_model_filename = natsort.natsorted(filenames)[-1]
        print("Done")

        # load and combine models:
        # models.load_model(model_filename, backbone_name=args.backbone)
        all_models = [models.load_model(most_recent_snapshot, backbone_name=backbone),
                      models.load_model(combine_model_filename, backbone_name=backbone)]

        return all_models
    else:
        return most_recent_snapshot

# %%
# Create the dual-memory model:

def CreateDualMemory(historical_snapshots_folder, snapshots_folder):
    print("\n" + "Started dual-memory modelling, looking in", historical_snapshots_folder)

    # If there is no "most_recent_snapshot", return None:

    most_recent_snapshot = LoadMostRecentModel(historical_snapshots_folder)
    if most_recent_snapshot is None:
        return None

    print("Most recent snapshot:", most_recent_snapshot)

    # Search folder for a Day-10 snapshot:
    # f"historical_snapshots/Day{day_number}/snapshots/"

    match = re.search("Day(\d+)", most_recent_snapshot)
    if match is None:
        raise ValueError("Filename doesn't conform to standard")

    day_number = int(match.group(1))
    print("Day number is:", day_number)

    if day_number > 10:
        print("Day number is greater than 10")
        find_day = 0
        # find_day = day_number - 10
        folder = os.path.join(historical_snapshots_folder, f"Day{find_day}/")
        filenames = glob(os.path.join(folder, "*"))
        combine_model_filename = natsort.natsorted(filenames)[-1]
        print("Done")

        # Load and combine models:

        new_model = CombineModels(combine_model_filename, most_recent_snapshot)
        output_filename = os.path.join(snapshots_folder, "resnet50_csv_0.h5")
        new_model.save(output_filename)
        return output_filename
    else:
        return most_recent_snapshot

# %%
class Arguments:
    pass

def LoadConfigArguments(config_filename):
    with open(config_filename) as inf:
        parameters = json.load(inf)

    args = Arguments()
    for key, value in parameters.items():
        setattr(args, key, value)  # args.key = value
    return args

# %%
# Train model:

def TrainModel(train_filename, train_config_filename):
    # Load default arguments in the config file:

    args = LoadConfigArguments(train_config_filename)

    # Train on top of previously trained model if it exists in snapshot path:
    if args.regularisation != 3.0:
        print("Running training with regularisation")
        pass
    
    if args.continual_learning_model is None:
        print("\n" + "Run model without dual memory")
        most_recent_snapshot = LoadMostRecentModel(args.historical_snapshots_folder)
        if most_recent_snapshot:
            print("Continuing training from", most_recent_snapshot)
            args.model = most_recent_snapshot
        else:
            print("Begin training from loaded weights")
            args.model = None
    
    if args.continual_learning_model == "dual_memory":
        print("\n" + "Run model with dual memory")
        most_recent_snapshot = LoadMostRecentModel(args.historical_snapshots_folder)
        if most_recent_snapshot:
            print("Continuing training from", most_recent_snapshot)
            args.model = most_recent_snapshot
        else:
            print("Begin training from loaded weights")
            args.model = None
    #     else:
    #         raise ValueError(f"Unknown continual learning mode: {args.continual_learning_model}")

    # If you only want to save the final snapshot and have it override each way use (and recode on line 181 in train.py):

    # if os.path.exists("snapshots\final_model_weights.h5"):
    #    args.snapshot = "snapshots\final_model_weights.h5"
    # else:
    #    args.snapshot = None

    args.annotations = train_filename

    # Run training script with train_config.json:

    train.main(args)

# %%
# Evaluate model:

def EvalModel(val_filename, day_number, val_config_filename):
    # Load default arguments in the config file:

    args = LoadConfigArguments(val_config_filename)

    # Evaluate last model in snapshot path:
    # filenames = glob.glob(os.path.join(args.snapshot_path, "*"))
    # if len(filenames) > 0:
    #     args.model = sorted(filenames)[-1]
    # else:
    #     args.model = None
    args.model = LoadMostRecentModel(args.historical_snapshots_folder)

    args.annotations = val_filename

    evaluate.main(args)


# %%
def GetMostRecentDayNumber(historical_snapshots_folder):
    subfolders = GetSubfolders(historical_snapshots_folder)
    if len(subfolders) == 0:
        return None
    most_recent_subfolder = natsort.natsorted(subfolders)[-1]
    print("\n" + "Most recent snapshot is", most_recent_subfolder)
    day_number_match = day_number = re.search("(\d+)", most_recent_subfolder)
    if day_number_match:
        day_number = day_number_match.group(1)
        return int(day_number)
    else:
        raise ValueError(f"Could not find day number in {most_recent_subfolder}")


# %%
# Training pipeline:

def Pipeline(day_number, train_config_filename, val_config_filename):  # Pipeline needs to now have a day number
    train_filename = f"data/train/Day{day_number}.csv"
    print("Training will resume with", train_filename)
    val_filename = f"data/val/Day{day_number}.csv"
    print("Evaluation will resume with", val_filename)
    start = timer()
    TrainModel(train_filename, train_config_filename)
    args = LoadConfigArguments(train_config_filename)
    MoveSnapshots(args.snapshot_path, day_number, args.historical_snapshots_folder)
    DeleteSnapshots(args.snapshot_path)
    EvalModel(val_filename, day_number, val_config_filename)
    end = timer()
    print("Day" + str({day_number}), "training duration was", (end-start)/60, "minutes")
    with open(args.time_savepath, 'a') as outf:
        outf.write(f"{day_number}, {(end-start)/60}" + "\n")


# %%
def RunPipeline(repeat, train_config_filename, val_config_filename):
    for i in range(repeat):
        print("\n" + "Repetition " + str(i + 1) + " out of " + str(repeat))
        args = LoadConfigArguments(
            train_config_filename) 
        day_number = GetMostRecentDayNumber(args.historical_snapshots_folder)
        if day_number is None:
            day_number = 1
        else:
            day_number += 1
        Pipeline(day_number, train_config_filename, val_config_filename)
        
# %%
RunPipeline(int(args.num_repeat), train_config_filename, val_config_filename)

# %%
# Alarm that sounds when training pipeline is finished:

duration = 5000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)