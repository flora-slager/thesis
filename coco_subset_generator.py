# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import required packages

from PIL import Image
import glob
import os
import json
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from timeit import default_timer as timer
import argparse

# %%

parser = argparse.ArgumentParser(description = "Script to subset COCO images")
parser.add_argument("--annotations", help = "COCO image annotations")
parser.add_argument("--source", help = "Source directory for COCO images") 
parser.add_argument("--subset", help = "Subset directory")
   
args = parser.parse_args()

# %%
# Open json file downloaded from COCO as part of "annotations" (2017):

file = args.annotations
data = json.loads(open(file).read())
image_data = data["images"] # Create object of the "images" list
annotation_data = data["annotations"] # Create object of the "annotations" list

print("Data collection done")

# %%
# Define id_finder that extracts id corresponding to image filename:

def id_finder(json_object, name): # 
    for dict in json_object:
        if dict["file_name"] == name:
            return (dict["id"])

print("Functions defined")

# Create a list of image_id that have the category_id of 1 (person):

annotation_df = pd.DataFrame(annotation_data) # Create pandas dataframe of annotations data
person_images = annotation_df.loc[annotation_df["category_id"] == 1, ["image_id"]] # Subset for image_id when 
                                                                                   # category_id equals 1
person_list = person_images["image_id"].values.tolist()

print("List created")

# %%
start = timer()
for filename in glob.glob(args.source + "*.jpg"):
    base_name = os.path.splitext(os.path.basename(filename))[0] # Take only base filename w/o path
    file_name = base_name + ".jpg" # Add jpg string so that filename can be read by for loop for idnumbers
    id_number = id_finder(image_data, file_name)
    if id_number in person_list:
        img = Image.open(filename)
        img.save(args.subset + base_name + ".jpg")
end = timer()

print("Seconds:", end-start)
print("Minutes:", (end-start)/60)
print("Hours:", (end-start)/3600)