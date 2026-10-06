import ast
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics


def safe_eval_references(references_str: str):
    try:
        data = ast.literal_eval(references_str)
    except Exception:
        return references_str  # fallback

    return data

def clean_url(url: str) -> str:
    return url.strip().strip('"').strip("'")

def csv_reader(name: str) -> pd.DataFrame:
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / f"{name}.csv"

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open CSV file in {file_path}: {e}") from e

    def extract_commit_info(references_str: str) -> list[tuple[str | None, str | None, str | None]]:
        if not isinstance(references_str, str):
            return []

        results = []

        github_match = re.finditer(r"github\.com/([^/]+/[^/]+)/commit/([0-9a-fA-F]+)", references_str)
        for match in github_match:
            results.append(("github", match.group(1), match.group(2)))

        gitlab_match = re.finditer(r"gitlab\.com/([^/]+/[^/]+?)(?:/-)?/commit/([0-9a-fA-F]+)", references_str)
        for match in gitlab_match:
            results.append(("gitlab", match.group(1), match.group(2)))

        return results
    
    df["commits"] = df["references"].apply(extract_commit_info)
    df = df.explode("commits").dropna(subset=["commits"])
    df[["PLATFORM", "REPO_PATH", "P_COMMIT"]] = pd.DataFrame(df["commits"].tolist(), index=df.index)
    df = df.drop(columns=["commits"])

    df = df.rename(columns={
        "id":   "CVE",
        "cwes": "V_CLASSIFICATION",
    })
    
    df = df.drop_duplicates(subset=["CVE", "P_COMMIT"])

    for col in ["V_ID", "PLATFORM", "Defect Type", "Defect Qualifier", "# Files", "Filename"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[["V_ID", "PLATFORM", "REPO_PATH", "CVE", "V_CLASSIFICATION", "P_COMMIT",
             "Defect Type", "Defect Qualifier", "# Files", "Filename"]]

    df = df.dropna(subset=["P_COMMIT", "REPO_PATH"])
    return df

def excel_reader(name: str) -> pd.DataFrame:
    """Accesses a excel in the data folder and converts it to a pandas dataframe
    
    Args:
        name (str): The name of the excel file without extension
    
    Raises:
        RuntimeError: If it can't open the excel file
    
    Returns:
        pd.DataFrame: A pandas dataframe
    """
    
    # Accessing the location of the excel
    root_dir = Path(__file__).parent.parent.parent  # Get the root folder
    data_dir = root_dir / "data"                    # Joins with data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / f"{name}.xlsx"
    
    # Opening excel
    try:
        df = pd.read_excel(file_path, sheet_name=0, header=1, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"Failed to open excel file in {file_path}: {e}") from e
    
    desired_cols = ["V_ID", "Project", "CVE", "V_CLASSIFICATION", "P_COMMIT", "Defect Type", "Defect Qualifier", "# Files", "Filename"]
    df = df[desired_cols]
    
    return df

def create_crosstab(df_predicted: pd.DataFrame, df_real: pd.DataFrame, category: str) -> pd.DataFrame:
    """Creates a table with the frequency of each defect for each IA model and returns it.\n
    It also prints the table the Frequency Table and a Percent Table
    
    Args:
        df_predicted (pd.DataFrame): Dataframe with analysis of AI responses
        df_real (pd.DataFrame): Dataframe with human responses
        category (str): The category being analyzed
    
    Returns:
        pd.DataFrame: Cross tabulation with two factors
    """
    
    df = pd.crosstab(df_predicted[category], df_predicted["Model"])                   # Creates a table with the frequency of each defect for each model
    human_counts = df_real[df_real["P_COMMIT"].isin(df_predicted["Sha"])]    # Only gets the defects classification from the humans that were analyzed by the IA
    human_counts = human_counts[category].value_counts()                # Counting Human Data
    
    valid_idx = human_counts.index      # Allowed defects
    # Divides the dataframe in two dataframes: One with the allowed defects and the other with the strange ones
    df_valid = df.loc[df.index.isin(valid_idx)]
    df_other = df.loc[~df.index.isin(valid_idx)]
    
    # Sums the rest in a single line
    if not df_other.empty:
        df_other = pd.DataFrame(df_other.sum()).T   # Sums all the lines in the rest Dataframe, creating a Series, then it converts into a Dataframe and transposes it
        df_other.index = ["Other"]                  # Names the line to Other
        df_final = pd.concat([df_valid, df_other])  # Concatenates the df_valid with the df_other
    else:
        df_final = df_valid
        
    
    df_final["Human"] = human_counts.reindex(df_final.index, fill_value=0).astype(int)   # Adds "Human" column to the Dataframe
    df_final.index.name = category  # Define index name
    
    # Data in percent
    percent = ((df_final / df_final.sum())*100).round(2)
    
    print("\n\n\n")
    print("Frequency Table:")
    print(df_final)
    print("\nPercent Table:")
    print(percent)
    
    return df_final

def update_accuracy(dataframes: list[pd.DataFrame], df_real: pd.DataFrame, df_predicted: pd.DataFrame) -> None:
    """Updates the accuracy dataframes with the number of correct and incorrect matches for each IA model.

    Args:
        dataframes (list[pd.DataFrame]): List of three dataframes to update
        df_real (pd.DataFrame): DataFrame with human analysis
        df_predicted (pd.DataFrame): DataFrame with IA analysis
    """
    human_defects = [Counter(df_real["Defect Type"]), Counter(df_real["Defect Qualifier"]), Counter(zip(df_real["Defect Type"], df_real["Defect Qualifier"]))]
            
    for model_name, df_model in df_predicted.groupby("Model"):
        ia_defects = [Counter(df_model["Defect Type"]), Counter(df_model["Defect Qualifier"]), Counter(zip(df_model["Defect Type"], df_model["Defect Qualifier"]))]
        
        for idx, df in enumerate(dataframes):
            ia_correct = sum((ia_defects[idx] & human_defects[idx]).values())
            df.loc[model_name, "Correct"] += ia_correct
            df.loc[model_name, "Incorrect"] += sum((ia_defects[idx]).values()) - ia_correct

def count_matches(df_real: pd.DataFrame, df_predicted: pd.DataFrame) -> list[pd.DataFrame]:
    """Creates three dataframes with the accuracy of every IA model for defect type, defect qualifier and both combined.
    
    Args:
        df_real (pd.DataFrame): A dataframe with human analysis
        df_predicted (pd.DataFrame): A dataframe with IA analysis
    
    Returns:
        list[pd.DataFrame]: Three dataframes with the accuracy of every IA model for defect type, defect qualifier and both combined
    """
    dataframes = [pd.DataFrame(0, columns=["Correct", "Incorrect"], index=np.unique(df_predicted["Model"])) for _ in range(3)]
    
    for commit, df_real_commit in df_real.groupby("P_COMMIT"):
        df_predicted_commit = df_predicted[df_predicted["Sha"] == commit]
        
        if any(df_real_commit["Filename"].isnull()) or any(df_real_commit["# Files"] != 1):
            update_accuracy(dataframes, df_real_commit, df_predicted_commit)
        else:
            for file_name, df_real_file in df_real_commit.groupby("Filename"):
                df_predicted_file = df_predicted_commit[df_predicted_commit["File Name"] == file_name]
                update_accuracy(dataframes, df_real_file, df_predicted_file)
    
    for name, df in zip(["Type", "Qualifier", "Combined"], dataframes):
        df["Accuracy (%)"] = (df["Correct"] / (df["Correct"] + df["Incorrect"]) * 100).round(2)
        df.Name = f"Defect {name}"
    
    return dataframes

def create_confusion_matrix(df_real: pd.DataFrame, df_predicted: pd.DataFrame, category: str | list[str], only_one_classification: bool = False) -> pd.DataFrame:
    """Creates a confusion matrix comparing human and IA classifications for a given category.
    
    Args:
        df_real (pd.DataFrame): DataFrame with human analysis
        df_predicted (pd.DataFrame): DataFrame with IA analysis
        category (str | list[str]): The category or categories being analyzed
    
    Returns:
        pd.DataFrame: A confusion matrix DataFrame
    """
    
    # Combine categories into one label if needed
    if isinstance(category, str):
        category = [category]
    
    if len(category) == 1:
        combined = category[0]
    else:
        combined = "_".join(category)
        df_real[combined] = df_real[category].astype(str).agg("_".join, axis=1)
        df_predicted[combined] = df_predicted[category].astype(str).agg("_".join, axis=1)
    
    # Determine valid labels
    possible_labels = sorted(df_real[combined].dropna().unique())
    
    all_metrics = []
    all_cf = []
    for ia_model in df_predicted["Model"].unique():
        df_model = df_predicted[df_predicted["Model"] == ia_model]
        
        # Lists for building the "flattened" actual and predicted labels
        actual = []
        predicted = []

        # Iterate per commit
        for commit, df_real_commit in df_real.groupby("P_COMMIT"):
            df_pred_commit = df_model[df_model["Sha"] == commit]

            if (only_one_classification):
                if len(df_real_commit) != 1:
                    continue
            
            # Convert to lists
            real_defects = df_real_commit[combined].tolist()
            pred_defects = df_pred_commit[combined].tolist()
            # Replace invalid labels in predicted
            pred_defects = [p if p in possible_labels else "Other" for p in pred_defects]

            # Matching predicted to actual
            temp_real = real_defects.copy()
            temp_pred = pred_defects.copy()
            for p in pred_defects:
                if p in temp_real:
                    actual.append(p)
                    predicted.append(p)
                    temp_real.remove(p)  # remove so it's matched only once
                    temp_pred.remove(p)  # remove so it's matched only once

            # Unmatched cases
            while temp_real or temp_pred:
                r = temp_real.pop(0) if temp_real else "Other"
                p = temp_pred.pop(0) if temp_pred else "Other"
                actual.append(r)
                predicted.append(p)
        
        # Replace invalid labels in predicted
        predicted = [p if p in possible_labels else "Other" for p in predicted]
        
        matrix_cf = metrics.confusion_matrix(
            actual,
            predicted,
            labels=possible_labels + ["Other"]
        )
        
        # Metrics
        accuracy = round(metrics.accuracy_score(actual, predicted), 2)
        precision = round(metrics.precision_score(actual, predicted, average='weighted', zero_division=0), 2)
        recall = round(metrics.recall_score(actual, predicted, average='weighted', zero_division=0), 2)
        f1_score = round(metrics.f1_score(actual, predicted, average='weighted', zero_division=0), 2)
        metrics_dict = {
            "Model": ia_model,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall/Sensitivity": recall,
            "F1_score": f1_score
        }

        
        all_metrics.append(metrics_dict)
        
        df_cf = pd.DataFrame(matrix_cf, index=possible_labels + ["Other"], columns=possible_labels + ["Other"])
        all_cf.append((ia_model, df_cf))
    
    # Create directories and save metrics and confusion matrices
    root_dir = Path(__file__).parent.parent.parent  # Get the root folder
    data_dir = root_dir / "data"                    # Joins with data directory
    metrics_dir = data_dir / "metrics"
    cf_dir = data_dir / "confusion_matrices"
    if (only_one_classification):
        metrics_dir = metrics_dir / "unique"
        cf_dir = cf_dir / "unique"
    else:
        metrics_dir = metrics_dir / "non_unique"
        cf_dir = cf_dir / "non_unique"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cf_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{combined}.csv"
    cf_path = cf_dir / f"{combined}_confusion_matrices.txt"

    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    
    with cf_path.open("w", encoding="utf-8") as f:
        for ia_model, df_cf in all_cf:
            f.write(f"Confusion Matrix for {ia_model}\n")
            f.write(df_cf.to_string())
            f.write("\n\n")