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
DATA_FOLDER = "/kaggle/tmp"

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

# Authenticate and upload via Kaggle API
try:
    # Initialize and authenticate the API client
    api = KaggleApi()
    api.authenticate()
    
    print("Starting dataset creation on Kaggle...")
    
    # Create the dataset on Kaggle from the specified folder
    # public=True creates a public dataset. Set to False for a private one.
    api.dataset_create_new(folder=DATA_FOLDER, public=True)
    
    print(f"Success! Your dataset is uploading to: https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_SLUG}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Double check that your '~/.kaggle/kaggle.json' file is in place and your username is correct.")
