# data/download_pdebench.py

import os
import ssl
import urllib.request

URL = "https://darus.uni-stuttgart.de/api/access/datafile/133020"
FILENAME = "1D_diff-sorp_NA_NA.h5"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, FILENAME)

    if os.path.exists(filepath):
        print("Dataset already exists:", filepath)
        return

    print("Downloading PDEBench dataset...")
    
    ssl._create_default_https_context = ssl._create_unverified_context
    
    urllib.request.urlretrieve(URL, filepath)
    
    print("Download complete.")


if __name__ == "__main__":
    main()
