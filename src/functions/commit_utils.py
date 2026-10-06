import csv
import os
from pathlib import Path
import time

from github import Commit, Github, GithubException, Repository
from gitlab import Gitlab
from gitlab.exceptions import GitlabGetError
from gitlab.v4.objects import Project, ProjectCommit

from dataclasses import dataclass

import requests


# main.py 
def fetch_github_commit(repo_url: str, sha: str, g: Github, repo_cache: dict[str, Repository.Repository]) -> Commit.Commit | None:
    """Fetches a commit given a full GitHub repo path (e.g. 'argoproj/argo-cd') and SHA."""
    try:
        if repo_url not in repo_cache:
            repo_cache[repo_url] = g.get_repo(repo_url)
        return repo_cache[repo_url].get_commit(sha)
    except GithubException as e:
        print(f"Error accessing '{repo_url}' with commit '{sha}': {e}")
        return None


def fetch_gitlab_commit(project: str, sha: str, gl: Gitlab, repo_cache: dict[str, Project]) -> ProjectCommit | None:
    """Given a project and a sha from a commit, fetches the commit from Gitlab """
    
    try:
        repo_name = project

        if repo_name not in repo_cache:
            repo_cache[repo_name] = gl.projects.get(repo_name)

        commit = repo_cache[repo_name].commits.get(sha)
        return commit
    except GitlabGetError as e:
        print(f"Error accessing repository '{project}' with commit '{sha}': {e}")       # If it can't access the repo or the commit, ir prints an error
        return None

@dataclass
class CommitFile:
    filename: str
    changes: int
    patch: str

def create_message(files: list[CommitFile], instruction: str) -> list[tuple[str, str]]:
    """"Given a commit and initial instruction, creates a prompt from the IA
    
    Args:
        commit (CommitFile): A commit from a repository, with its files, changes and patch
        instruction (str): An instruction for the IA that will join with the commit and a intended response format
        
    Returns:
        list[tuple[str, str]]: A list of tuples. Each tuple has a prompt and the name of the file that the prompt was created for
    """
    
    if files is None:
        return []
    
    response_format = "Your response should not provide an explanation and should only contain the following response format for each defect you classify in each file:\nDefect Type: <Defect Type>\nDefect Qualifier: <Defect Qualifier>"  
    # Save prompt for each file from the commit
    prompts = []
    for f in files:
        file_prompt = f"{instruction}\n\nFile name: {f.filename}\nChanges: {f.changes}\nPatch (diff):\n{f.patch}\n\n{response_format}"      # Joins the prompt with the commit and the format intended
        prompts.append((file_prompt, f.filename))
    
    return prompts

from openai import OpenAI

def call_ollama_endpoint(model: str, prompt: str) -> tuple[str, int, int]:
    """Calls the custom chat endpoint (ollama-style) and returns (content, prompt_tokens, response_tokens)"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2}
    }

    response = requests.post(
        os.getenv("CHAT_ENDPOINT"),
        json=payload,
        auth=requests.auth.HTTPBasicAuth(os.getenv("CHAT_API_NAME"), os.getenv("CHAT_API_PASSWORD")),
    )
    response.raise_for_status()

    data = response.json()
    content = data["message"]["content"]
    prompt_eval_count = data.get("prompt_eval_count", 0)
    response_eval_count = data.get("eval_count", 0)

    return content, prompt_eval_count, response_eval_count


def call_openai(model: str, prompt: str) -> tuple[str, int, int]:
    """Calls the OpenAI API and returns (content, prompt_tokens, response_tokens)"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )

    content = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    return content, prompt_tokens, completion_tokens


def call_model(provider: str, model: str, prompt: str, folder: Path) -> None:
    """Calls an IA model via the given provider, runs the specified prompt and stores the response in a text file
    
    Args:
        provider (str): The AI provider to use ('ollama' or 'openai')
        model (str): The name of the IA model that will be run
        prompt (str): The message that will be given to the IA
        folder (Path): The folder where the text file will be stored in
    """
    
    model_name: str = model.partition(":")[0]       # Take model name before ':' if present
    file_path: Path = folder / f"{model_name}.txt"
    metrics_path: Path = folder / "metrics.csv"
    
    if file_path.exists():
        return
    
    try:
        start = time.perf_counter()

        if provider == "ollama":
            content, prompt_eval_count, response_eval_count = call_ollama_endpoint(model, prompt)
        elif provider == "openai":
            content, prompt_eval_count, response_eval_count = call_openai(model, prompt)
        else:
            raise ValueError(f"Provider not found: {provider}")

        elapsed = time.perf_counter() - start
        
        file_path.write_text(content, encoding="utf-8")
        
        write_header = not metrics_path.exists()
        with open(metrics_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Model", "Elapsed Time (seconds)", "Prompt Tokens", "Response Tokens", "Total Tokens"])
            
            writer.writerow([
                model, 
                round(elapsed, 3),
                prompt_eval_count, 
                response_eval_count, 
                prompt_eval_count + response_eval_count
            ])
            
    except Exception as e:
        print(f"Error calling model {model} or writing file {file_path}: {e}")
    
def normalize_github_files(commit: Commit.Commit) -> list[CommitFile]:
    if commit is None:
        return []
    
    return [
        CommitFile(
            filename=f.filename,
            changes=f.changes,
            patch=f.patch or ""
        )
        for f in commit.files
    ]

from gitlab.exceptions import GitlabGetError

def normalize_gitlab_files(commit: ProjectCommit) -> list[CommitFile]:
    if commit is None:
        return []
    
    try:
        # Tentamos ir buscar o diff
        diffs = commit.diff()
    except Exception as e:
        print(f"\n[AVISO] Falha ao extrair diff do commit GitLab '{commit.id}': {e}")
        return []

    return [
        CommitFile(
            filename=f["new_path"],
            # Prevenção: caso f["diff"] venha a None, não dá erro ao contar os "\n"
            changes=f["diff"].count("\n") if f.get("diff") else 0,
            patch=f.get("diff") or ""
        )
        for f in diffs
    ]

def process_commit(row, provider: str, prompt: str, models: list[str], g: Github, gl: Gitlab, repo_cache: dict[str, Repository.Repository | Project], run_id: str) -> None:

    root_dir = Path(__file__).parent.parent.parent  # Get the root folder
    output_dir = root_dir / "trustdev-output" / run_id  # Joins with output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = output_dir / row.REPO_PATH.replace("/", "-")   # Creates a directory for the repository, replacing '/' with '-' to avoid issues in folder names
    repo_dir.mkdir(parents=True, exist_ok=True)

    sha: str = row.P_COMMIT
    sha_dir: Path = repo_dir / sha            # Directory's path to save IA's response
    sha_dir.mkdir(parents=True, exist_ok=True)  # Creates the directory if it doesn't exist; parents=True creates every needed parent directory if it doesn't exist; exist_ok=True doesn't give a error if the directory already exists
    
    is_github = row.PLATFORM == "github"
    is_gitlab = row.PLATFORM == "gitlab"
    
    # Create prompt for the IA
    if is_github:
        commit: Commit.Commit = fetch_github_commit(row.REPO_PATH, sha, g, repo_cache)
        if commit is None:
            print(f"Commit '{sha}' not found in GitHub repository '{row.REPO_PATH}'")
            return
        files = normalize_github_files(commit)
    elif is_gitlab:
        commit: ProjectCommit = fetch_gitlab_commit(row.REPO_PATH, sha, gl, repo_cache)
        if commit is None:
            print(f"Commit '{sha}' not found in GitLab repository '{row.REPO_PATH}'")
            return
        files = normalize_gitlab_files(commit)
    else:
        raise ValueError("Unsupported URL format")

    content = create_message(files, prompt)
    
    for message, file_name in content:
        safe_name = file_name.replace("/", "-").replace(".", "_")
        file_dir: Path = sha_dir / safe_name
        file_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            call_model(provider, model, message, file_dir)