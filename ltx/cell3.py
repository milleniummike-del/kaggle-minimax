import os
print("Running cell3.py")
import json
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# 1. Define configuration variables
# Note: The 'id' must follow the format 'your-kaggle-username/your-desired-dataset-slug'
KAGGLE_USERNAME = "mikegriffin2027"  
DATASET_SLUG = "ltx-23-models"
DATASET_TITLE = "My Awesome Python Dataset"
DATA_FOLDER = "/kaggle/tmp/models"

# Programmatically create the required 'dataset-metadata.json' file
metadata = {
    "title": DATASET_TITLE,
    "id": f"{KAGGLE_USERNAME}/{DATASET_SLUG}",
    "licenses": [
        {
            "name": "CC0-1.0"  # Options include: CC0-1.0, CC-BY-SA-4.0, gpl-3.0, etc.
        }
    ]
}

metadata_path = os.path.join(DATA_FOLDER, "dataset-metadata.json")
with open(metadata_path, "w") as json_file:
    json.dump(metadata, json_file, indent=4)

print(f"Created metadata file at: {metadata_path}")
print("kaggle datasets create -p <FOLDER_PATH> -r zip")
