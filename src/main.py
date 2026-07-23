import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from dotenv import load_dotenv
from github import Auth, Github, Repository
from gitlab import Gitlab
from gitlab.v4.objects import Project
from tqdm import tqdm

from functions.commit_utils import process_commit
from functions.data_utils import excel_reader, csv_reader

from datetime import datetime
from pathlib import Path

prompt = "A defect type can be one of the following categories: 1) Assignment/Initialization: a problem related to an assignment of a variable or no assignment at all; 2) Checking: a problem with conditional logic (e.g., condition in a if-clause or in a loop); 3) Timing: a problem with serialization of shared resources; 4) Algorithm/Method: a problem with implementation that does not require a design change to be fixed; 5) Function: a problem that needs a reasonable amount of code to be fixed due to incorrect implementation or no implementation at all; 6) Interface: a problem in the interaction between components (e.g., parameter list). With this in mind, what’s the defect type of the orthogonal defect classification (ODC) in the following commit? On the other hand, a defect qualifier can be one of the following categories: 1) Missing: new code needs to be added to fix the defect; 2) Incorrect: the code is incorrectly implemented and needs adjustment to fix the defect; 3) Extraneous: unnecessary. With that in mind, what’s the defect type of the orthogonal defect classification (ODC) in the following commit? With this in mind, what’s the defect type and defect qualifier of the orthogonal defect classification (ODC) in the following commit?"

# IA Models Used
models = [
    "qwen3:1.7b",
    "deepseek-coder-v2:latest",
    "qwen2.5-coder:14b",
    "gpt-oss:20b",
    "llama3.1:70b",
    "gemma3:27b",
]

df_real = csv_reader("cves_merged")
repo_cache: dict[str, Repository.Repository | Project] = {}

base_folder = Path("/hdd/josep/tdsai-odc-output")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

run_output_dir = base_folder / timestamp
run_output_dir.mkdir(parents=True, exist_ok=True)

load_dotenv()
token = os.getenv("GITHUB_TOKEN")
if token is None:
    raise RuntimeError("GITHUB_TOKEN environment variable not set.")
auth = Auth.Token(token)
g = Github(auth=auth)
gl = Gitlab()

executor_func = partial(process_commit, prompt=prompt, models=models, g=g, gl=gl, repo_cache=repo_cache, output_dir=run_output_dir)

with ThreadPoolExecutor(max_workers=5) as executor:
    list(tqdm(executor.map(executor_func, df_real.itertuples(index=False)), total=len(df_real), desc="Processing commits", unit=" commits"))