from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from functions.regex_utils import extract_defects

DEFECT_TYPE_MAP = {
    "assignment": "Assignment/Initialization",
    "initialization": "Assignment/Initialization",
    "assignment/initialization": "Assignment/Initialization",
    "algorithm": "Algorithm/Method",
    "method": "Algorithm/Method",
    "algorithm/method": "Algorithm/Method",
    "checking": "Checking",
    "function": "Function",
    "interface": "Interface",
    "timing": "Timing",
}
DEFECT_QUALIFIER_MAP = {
    "missing": "Missing",
    "incorrect": "Incorrect",
    "extraneous": "Extraneous",
}

def bar_graph(data: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data.index, data.values)
    plt.bar_label(bars, padding=3, fontsize=8)  # Add labels to the bars
    plt.xlabel("Model")
    plt.ylabel("Count")
    plt.title(title)

folder = Path("trustdev-output")

rows = []

for run_folder in folder.iterdir():
    if not run_folder.is_dir():
        continue

    for project_folder in run_folder.iterdir():    # For every project folder in the main folder
        if not project_folder.is_dir():
            continue

        files = list(project_folder.rglob("*.txt"))

        for file_path in files:     # For every text file in the main folder, including subfolders
            try:
                text = file_path.read_text(encoding="utf-8")    # pathlib method that reads the file and returns a string
            except (OSError, PermissionError, UnicodeDecodeError) as e:      # If there's an error with the path or decoding, it continues
                print(f"Error reading {file_path}: {e}")
                continue
            
            defects = extract_defects(text)
            
            for defect in defects:
                normalized_type = DEFECT_TYPE_MAP.get(
                    defect[0].lower() if defect[0] else "", "Unknown"
                )
                normalized_qualifier = DEFECT_QUALIFIER_MAP.get(
                    defect[1].lower() if defect[1] else "", "Unknown"
                )

                rows.append({
                    "Run": run_folder.name,
                    "Project": file_path.parts[-4],
                    "Sha": file_path.parts[-3],
                    "File Name": file_path.parts[-2],
                    "Model": file_path.stem,
                    "Defect Type": normalized_type,
                    "Defect Qualifier": normalized_qualifier
                })

data = pd.DataFrame(rows)

# Counts the number of files processed by each model
unique_file_models = data[["File Name", "Model"]].drop_duplicates()
model_counts = unique_file_models["Model"].value_counts()
bar_graph(model_counts, "Number of Files Processed per Model")

# Counts the number of defects identified by each model
model_counts = data["Model"].value_counts()
bar_graph(model_counts, "Number of Defects Identified per Model")

# Creates a crosstab for each model, showing the counts of defect types and defect qualifiers
for model in data["Model"].unique():
    model_data = data[data["Model"] == model]
    
    table = pd.crosstab(index=model_data["Defect Qualifier"], columns=model_data["Defect Type"], margins=True, margins_name="Total")

    with open(f"output/defect_crosstab/{model}.csv", "w", encoding="utf-8", newline="") as f:
        table.to_csv(f)

# Creates a comparison table for each file, showing the defects identified by each model
data["Defect"] = data["Defect Type"] + " - " + data["Defect Qualifier"]
grouped = data.groupby(["Project", "Sha", "File Name", "Model"])["Defect"].agg(lambda x: sorted(set(x))).reset_index()
comparison_df = grouped.pivot(
    index=["Project", "Sha", "File Name"], 
    columns="Model",
    values="Defect"
).reset_index()

with open("output/defect_comparison.csv", "w", encoding="utf-8", newline="") as f:
    comparison_df.to_csv(f, index=False)

plt.show()