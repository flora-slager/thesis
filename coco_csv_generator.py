# %%
import argparse
import glob
import json
import numpy as np
import os
import pandas as pd
import random
import re

# %%
parser = argparse.ArgumentParser(description = "Script to generate CSV files.")
parser.add_argument("--path_to_images", help = "Directory containing training or validaion image subsets.")
parser.add_argument("--num_files", type = int, help = "Number of images within each CSV.") 
parser.add_argument("--path_to_json", help = "Path to COCO annotation file.") 
parser.add_argument("--csv_save_path", help = "Directory within which to save the CSV files generated.")  
parser.add_argument("--files", type = int, help = "Number of CSV files to generate") 
parser.add_argument("--tuning", help = "Specify whether dataset is for tuning or not.")  
    
args = parser.parse_args()

# %%
# ## USER INPUT ##

# # Specify paths to training and validation data:

# train_img = r"C:\Users\GGPC\coco_data\trainpeople\\" # Can hard coded into coco_csv
# train_json = r"C:\Users\GGPC\coco_data\annotations\instances_train2017.json" # Can be hard coded into coco_csv

# val_img = r"C:\Users\GGPC\coco_data\valpeople\\" # Can hard coded into coco_csv
# val_json = r"C:\Users\GGPC\coco_data\annotations\instances_val2017.json" # Can be hard coded into coco_csv

# #csv_save_path = r"C:\keras_retinanet\keras_retinanet\data\train\{i}.csv"

# # Specify training sample size: 

# train_files = 9000
# val_files = 384 # Should be 20% of training set size


# %%
def ImageSelector(path_to_images, num_files = int):
    """
    Extracts basenames of a random sample of image files in a specified folder into a list. 
    To select non-randomly, replace first line with:
    for filename in glob.glob(folder + "\*.jpg")[:file_range]:
    
    Note: At this stage, this function can only handle paths that have no other numbers in them outside of the file basenames.
    
    Arguments:
        path_to_images{str}: Path to image folder.
        num_files{int}: Subset size for selection.
    
    Returns:
        image_basenames{list}: List of image basenames for specified subset.
    
    """
    random_sample = random.sample(glob.glob(path_to_images + "\*.jpg"), num_files)
    seperator = ", "
    subset_as_string = seperator.join(random_sample)
    image_basenames = re.findall("[0-9]+", subset_as_string)
    return image_basenames


# %%
# Open, read and load json file:

def jsonExtractor(path_to_json):
    """
    Opens, reads and loads specified (COCO) json file.
    
    Arguments:
        path_to_json{path}: Path to (COCO annotation) json file.
        
    Returns:
        master{json}: Loaded (COCO annotation) json file.
    """    
    loaded_json = json.loads(open(path_to_json).read())
    return loaded_json


# %%
# Add full file path in filename column:

def FilePath(filename, path_to_images):
    """
    Converts filename column to including the full paths for each filename.
    
    Arguments:
        filename{str}: Image filename.
        base_path{str}: Path to image folder; same as that used in ImageSelector.
        
    Returns:
        full_path{str}: Full path of each filename
    """
    full_path = path_to_images + filename
    return full_path


# %%

# Include file basename as new variable in dataframe:

def Basename(filename):
    """
    Adds a column of file basenames as numerical values. 
    These are used in conjunction with the output of ImageSelector to subset the dataframe.
    
    Arguments:
        filename{str}: Image filename.
        
    Returns:
        basename{num}: Image file basename.
    
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    return basename


# %%
# Clean images data for use in csv:

def ImagesClean(coco_json, path_to_images):
    """
    Converts image json data to a Pandas dataframe.
    Subsets for columns required in final csv.
    Cleans dataframe.
    
    Arguments:
        coco_json{json}: Loaded COCO annotation json data
        
    Returns:
         images_df{pd}: Pandas dataframe of image data subsetted for required columns
    """
    images = coco_json["images"]
    images_df = pd.DataFrame(images) # Create pandas dataframe of images json data
    
    images_df = images_df.loc[:, ["file_name", "width", "height", "id"]] # Override images_df with a subset of columns
    images_df = images_df.rename(columns = {"file_name":"filename", "id":"image_id"}) # Change "id" column to "image_id" for merging and change "file_name" to "filename" for consistency
    images_df["filename"] = images_df["filename"].apply(FilePath, args=[path_to_images]) # Add full path to filenames in filename column
    images_df["basename"] = images_df["filename"].apply(Basename) # Apply basename column to dataframe
    
    return images_df


# %%
def ConvertCOCO(coco_json):
    """
    Extracts COCO bounding box coordinates from annotation json and converts them to required x1, y1, x2, y2 format
    
    Arguments:
        coco_json{json}: Loaded COCO annotation json data.
    
    Returns:
        annotations_df{pd}: Pandas dataframe of annotations data with bounding box coordinates in required format.
    """
    annotations = coco_json["annotations"]
    annotations_df = pd.DataFrame(annotations) # Create pandas dataframe of annotations json data
    
    annotations_df[["x", "y", "width", "height"]] = pd.DataFrame(annotations_df.bbox.tolist(), index = annotations_df.index) # Expand "bbox" column into required coordinate columns
    annotations_df["x1"] = annotations_df["x"]
    annotations_df["y1"] = annotations_df["y"]
    annotations_df["x2"] = annotations_df["x"] + annotations_df["width"]
    annotations_df["y2"] = annotations_df["y"] + annotations_df["height"]
    
    return annotations_df


# %%
# Clean annotations_df dataframe:

def AnnotationsClean(annotations_df):
    """
    Cleans columns in annotations dataframe as required in final csv.
    
    Arguments:
        annotations_df{pd}: Pandas dataframe of annotations data with bounding box coordinates in required format.
        
    Returns:
         annotations_df{pd}: Clean Pandas dataframe of annotations data.
    """
    annotations_df = annotations_df.rename(columns = {"category_id":"class"}) # Change "category_id" column to "class" for consistency
    annotations_df = annotations_df.loc[:, ["image_id", "x1", "y1", "x2", "y2", "class"]] # Override annotations_df with subsetted columns
    annotations_df[["x1", "y1", "x2", "y2"]] = annotations_df[["x1", "y1", "x2", "y2"]].astype("int") # Convert float coordinates to integers
    return annotations_df


# %%
# Merge image and annotations dataframes on "image_id":

def Merge(images_df, annotations_df):
    """
    Merges images and annotations dataframes.
    
    Arguments:
        images_df{pd}: Pandas dataframe of image data subsetted for required columns.
        annotations_df{pd}: Pandas dataframe of annotations data subsetted for required columns.
    
    Returns:
        merged_df{pd}: Pandas dataframe of merged data subsetted for required columns.
    """
    merged_df = pd.merge(left = images_df, right = annotations_df, on = "image_id") # Merge images_df and annotations_df
    merged_df = merged_df.loc[:, ["filename", "x1", "y1", "x2", "y2", "class", "basename"]] # Specify columns required and override train2017 with subset
    return merged_df


# %%
# Subset merged dataframe for class_name = 1 and replace by label (person):

def ClassSubsetPerson(merged_df):
    """
    Subsets Pandas dataframe on class column to include only class = 1 and replaces values in same column with "person".
    Note: To avoid error or warning when using .replace will make a copy of input dataframe.
    
    Arguments:
        merged_df{pd}: Pandas dataframe of merged data subsetted for required columns.
    
    Returns:
        retinanet_df{pd}: Pandas dataframe suitable for Keras-Retinanet input.
    """    
    retinanet_df = merged_df.copy()
    retinanet_df = retinanet_df.loc[(merged_df["class"] == 1), :]
    retinanet_df["class"] = retinanet_df["class"].replace({1: "person"})
    return retinanet_df


# %%
# Subset merged dataframe for class_name = 1 and other classes and replace by labels:

def ClassSubsetCombined(merged_df):
    """
    Subsets Pandas dataframe on class column to include only class = 1 and replaces values in same column with "person".
    Note: To avoid error or warning when using .replace will make a copy of input dataframe.
    
    Arguments:
        merged_df{pd}: Pandas dataframe of merged data subsetted for required columns.
    
    Returns:
        retinanet_df{pd}: Pandas dataframe suitable for Keras-Retinanet input.
    """    
    array = [1, 3, 15, 18, 27, 37, 47, 54, 65, 72, 82, 84]
    retinanet_df = merged_df.copy()
    retinanet_df = merged_df.loc[merged_df['class'].isin(array)]
    retinanet_df["class"] = retinanet_df["class"].replace({1: "person", 3: "car", 15: "bench", 18: "dog", 27: "backpack", 37: "sports ball", 47: "cup", 54: "sandwich", 65: "bed", 72: "tv", 82: "refrigerator", 84: "book"})
    return retinanet_df

# %%
# Subset dataframe according to to the random sample of images specified with ImageSelector:

def RandomSubset(retinanet_df, random_subset):
    """
    Subsets Pandas dataframe to include only rows within random sample of image basenames specified by ImageSelector.
    
    Arguments:
        retinanet_df{pd}: Pandas dataframe suitable for Keras-Retinanet input.
        random_subset{list}: List of image basenames randomly samples via ImageSelector.
    
    Returns:
        csv_df{pd}: Pandas dataframe ready for conversion to csv.
    """
    retinanet_df = retinanet_df.loc[retinanet_df.basename.isin(random_subset)]
    csv_df = retinanet_df.drop("basename", axis = 1) # Drop basename variable from dataframe
    return csv_df

# %%
# Save dataframe as csv file:

def SaveDataframe(df, csv_save_path = args.csv_save_path):
    df.to_csv(csv_save_path, index = False, header = False)
    print("Save successful... and don't you just look lovely today!")

# %%
def RunGenerator(path_to_images = args.path_to_images, num_files = args.num_files, path_to_json = args.path_to_json, csv_save_path = args.csv_save_path): 
    """
    Arguments:
        path_to_images{str}: Path to image folder.
        num_files{int}: Size of random sample.
        path_to_json{str}: Path to COCO annotation json; training or validation.
        csv_save_path{str}: Save path for csv, including intended csv name.
    """
    print("\n" + "Creating data...")
    random_subset = ImageSelector(path_to_images, num_files)
    master_json = jsonExtractor(path_to_json)
    clean_images = ImagesClean(master_json, path_to_images)
    bounding_box = ConvertCOCO(master_json)
    clean_annotations = AnnotationsClean(bounding_box)
    merged_df = Merge(clean_images, clean_annotations)
    all_df = ClassSubsetCombined(merged_df)
    all_csv = RandomSubset(all_df, random_subset)
    SaveDataframe(all_csv, csv_save_path)

def RunPeople(path_to_images = args.path_to_images, num_files = args.num_files, path_to_json = args.path_to_json, csv_save_path = args.csv_save_path):
    """
    Arguments:
        path_to_images{str}: Path to image folder.
        num_files{int}: Size of random sample.
        path_to_json{str}: Path to COCO annotation json; training or validation.
        csv_save_path{str}: Save path for csv, including intended csv name.
    """
    print("\n" + "Creating data...")
    random_subset = ImageSelector(path_to_images, num_files)
    master_json = jsonExtractor(path_to_json)
    clean_images = ImagesClean(master_json, path_to_images)
    bounding_box = ConvertCOCO(master_json)
    clean_annotations = AnnotationsClean(bounding_box)
    merged_df = Merge(clean_images, clean_annotations)
    retinanet_df = ClassSubsetPerson(merged_df)
    csv_df = RandomSubset(retinanet_df, random_subset)
    SaveDataframe(csv_df, csv_save_path)  

# %%
def DataGenerator(path_to_images = args.path_to_images, num_files = args.num_files, path_to_json = args.path_to_json, csv_save_path = args.csv_save_path, files = args.files, tuning = args.tuning):
    for i in range(files):
        print("Repetition " + str(i+1) + ' out of ' + str(files))
        date = "Day" + str(i + 1)
        csv_save_path = f"{args.csv_save_path}{date}.csv"
        if args.tuning == "person":
            RunPeople(path_to_images, num_files, path_to_json, csv_save_path)
        else:
            RunGenerator(path_to_images, num_files, path_to_json, csv_save_path)

# %%
DataGenerator(args.path_to_images, args.num_files, args.path_to_json, args.csv_save_path, args.files, args.tuning)
