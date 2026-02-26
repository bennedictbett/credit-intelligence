import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "creditiq"

list_of_files = [
    
    ".github/workflows/.gitkeep",


    f"src/{project_name}/__init__.py",
    "src/__init__.py",
    "src/data_processing.py",
    "src/feature_engineering.py",
    "src/train.py",
    "src/predict.py",
    "src/evaluate.py",

    
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",

    
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "notebooks/03_modeling.ipynb",
    "notebooks/04_model_evaluation.ipynb",
    "notebooks/05_explainability.ipynb",

    
    "models/.gitkeep",

    
    "app/main.py",
    "app/static/.gitkeep",
    "app/templates/.gitkeep",

    
    "tests/test_pipeline.py",

    
    "reports/figures/.gitkeep",
    "reports/summary.md",

    
    "config.yaml",

    
    "requirements.txt",
    "setup.py",
    "README.md",
    ".gitignore",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

logging.info("CreditIQ project structure created successfully!")