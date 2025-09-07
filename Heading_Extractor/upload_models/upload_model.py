# upload_model.py
from huggingface_hub import HfApi, create_repo, upload_folder, login
# optional: login(token="hf_xxx")  # or login from terminal once

api = HfApi()

# create a repo under your username (replace username and model-name)
#   api.create_repo(repo_id="itsGoodman/electra_small_heading_classifier", repo_type="model", private=False)

# upload the whole local folder to the new repo
api.upload_folder(folder_path=r"P:/Projects/PDF-Heading-Extraction/Heading_Extractor/ELECTRA/electra-small-heading-classifier-expanded",
                  repo_id="itsGoodman/electra_small_heading_classifier")