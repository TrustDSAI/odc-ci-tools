from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from functions.data_utils import (
    count_matches,
    create_confusion_matrix,
    create_crosstab,
    excel_reader,
)
from functions.graphs import create_bar, create_pie
from functions.regex_utils import extract_defects

folder = Path("trustdev-output")
files = list(folder.rglob("*.txt"))
data: list[dict[str, str | None]] = []


for project_folder in folder.iterdir():    # For every project folder in the main folder
    files = list(project_folder.rglob("*.txt"))
    for file_path in tqdm(files, desc="Processing files", unit=" files"):     # For every text file in the main folder, including subfolders
        try:
            text = file_path.read_text(encoding="utf-8")    # pathlib method that reads the file and returns a string
        except (OSError, PermissionError, UnicodeDecodeError) as e:      # If there's an error with the path or decoding, it continues
            print(f"Error reading {file_path}: {e}")
            continue
        
        defects = extract_defects(text)
        
        for defect in defects:
            defect_type, defect_qualifier = defect
            data.append({
                "Sha": file_path.parts[1],          # file_path.parts = ('responses', 'sha', 'file_name', 'model.txt')
                "File Name": file_path.parts[2],
                "Model": file_path.stem,            # Returns the stem (file name without extension)
                "Defect Type": defect_type, 
                "Defect Qualifier": defect_qualifier
                })


df_predicted = pd.DataFrame(data)          # Create DataFrame

script_dir = Path(__file__).parent      # Get the folder where this file is located (src/)
data_dir = script_dir.parent / "data"   # Goes up one level and joins with data folder
data_dir.mkdir(parents=True, exist_ok=True)
output_path = data_dir / "output.csv"

try:
    df_predicted.to_csv(output_path, index=False, encoding="utf-8")    # Export DataFrame to CSV
except (OSError, PermissionError, UnicodeEncodeError) as e:
    raise RuntimeError(f"Failed to write CSV to {output_path}: {e}") from e

df_real = excel_reader("vulnerabilities")

# Creating Bar Graphs
fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True, num="Bar Graph - Vulnerabilities", sharey=True)    # constrained_layout automatically adjusts the space between subplots, titles, labels and legends, removing all empty space
for i, defect in enumerate(["Defect Type", "Defect Qualifier"]):
    crosstab = create_crosstab(df_predicted, df_real, defect)
    create_bar(crosstab, axes[i])
    create_pie(crosstab)

for df in count_matches(df_real, df_predicted):
    print(f"\n=== {df.Name} Accuracy ===")
    print(df)

for category in ["Defect Type", "Defect Qualifier", ["Defect Type", "Defect Qualifier"]]:
    create_confusion_matrix(df_real, df_predicted, category)
    create_confusion_matrix(df_real, df_predicted, category, only_one_classification=True)

plt.show()