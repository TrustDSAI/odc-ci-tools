import csv
import os
from pathlib import Path
import time

from github import Commit, Github, GithubException, Repository
from gitlab import Gitlab
from gitlab.exceptions import GitlabGetError
from gitlab.v4.objects import Project, ProjectCommit

from dataclasses import dataclass

from openai import OpenAI
import google.generativeai as genai

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

def call_ollama(model_name: str, prompt: str) -> tuple[str, int, int]:
    """Helper function to call ollama"""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2}
    }
    
    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/chat")
    response = requests.post(endpoint, json=payload)
    response.raise_for_status()
    
    data = response.json()
    content = data["message"]["content"]
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    
    return content, prompt_tokens, completion_tokens

def call_openai(model_name: str, prompt: str) -> tuple[str, int, int]:
    """Helper function to call openai API"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=False
    )
    
    content = response.choices[0].message.content
    # Metrics extraction
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    
    return content, prompt_tokens, completion_tokens

def call_gemini(model_name: str, prompt: str) -> tuple[str, int, int]:
    """Helper function to call gemini API"""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Temperatura defining
    config = genai.types.GenerationConfig(temperature=0.2)
    model = genai.GenerativeModel(model_name)
    
    response = model.generate_content(prompt, generation_config=config)
    
    content = response.text
    # Metrics extraction
    prompt_tokens = response.usage_metadata.prompt_token_count
    completion_tokens = response.usage_metadata.candidates_token_count
    
    return content, prompt_tokens, completion_tokens

def call_model(provider: str, prompt: str, folder: Path, model: str | None = None) -> None:
    """Calls IA model via determinated provider, runs the specified prompt and stores the response in a text file
    
    Args:
        provider (str): The AI provider (for now, 'openai', 'gemini', 'ollama')
        prompt (str): The message that will be given to the IA
        folder (Path): The folder where the text file will be stored in
        model (str): The name of the IA model that will be run.
    """
    
    # If model is none, we use the provider (just because ollama needs)
    model_name = model if model else provider
    
    file_path: Path = folder / f"{model_name}.txt"
    metrics_path: Path = folder / "metrics.csv"
    
    if file_path.exists():
        return
    
    try:
        start = time.perf_counter()
        
        if provider == "openai":
            content, prompt_eval_count, response_eval_count = call_openai(model_name, prompt)
        elif provider == "gemini":
            content, prompt_eval_count, response_eval_count = call_gemini(model_name, prompt)
        elif provider == "ollama":
            content, prompt_eval_count, response_eval_count = call_ollama(model_name, prompt)
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
                model_name, 
                round(elapsed, 3),
                prompt_eval_count, 
                response_eval_count, 
                prompt_eval_count + response_eval_count
            ])
            
    except Exception as e:
        print(f"Error calling model {model_name} or writing file {file_path}: {e}")

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

def normalize_gitlab_files(commit: ProjectCommit) -> list[CommitFile]:
    if commit is None:
        return []
    
    return [
        CommitFile(
            filename=f["new_path"],
            changes=f["diff"].count("\n"),  # aproximação, gitlab não dá changes direto
            patch=f["diff"] or ""
        )
        for f in commit.diff()
    ]

def process_commit(row, prompt: str, models: list[str], g: Github, gl: Gitlab, repo_cache: dict[str, Repository.Repository | Project]) -> None:

    root_dir = Path(__file__).parent.parent.parent  # Get the root folder
    output_dir = root_dir / "trustdev-output"                # Joins with output directory
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
        # If we want to use a provider like openAI or genAi, we just have to define provider as "openai" or "gemini"
        # If we want to use ollama as our provider, provider should be ollama and model should be the model to use.
        provider = "ollama"
        #provider = "gemini"
        #provider = "openai"
        for model in models:
            call_model(
                provider=provider,
                prompt=message,
                folder=file_dir,
                model=model,
                )