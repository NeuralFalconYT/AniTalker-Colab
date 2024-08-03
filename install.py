import subprocess
import sys
import os
import requests
import re
def torch_command():
    try:
        # Run the nvcc command
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        # Check if the command was successful
        if result.returncode == 0:
            # Extract version information from the output
            lines = result.stdout.split('\n')
            versions = []
            for line in lines:
                if 'release' in line:
                    # Find the release line and extract the version
                    version_info = line.split('release')[-1].strip()
                    # Use regex to extract the version in the format X.Y
                    match = re.search(r'(\d+\.\d+)', version_info)
                    if match:
                        versions.append(match.group(1))
            # Return the most recent version if multiple are found
            if versions:
                # Format version as required by the pip command
                cuda_version=sorted(set(versions), reverse=True)[0]
                print(f"Cuda version: {cuda_version}")
                cuda_version=cuda_version.replace('.', '')
                pip_command=f"pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu{cuda_version}"
                return pip_command
            else:
                return "CUDA version not found in output."
        else:
            return "nvcc command not found or failed to execute."
    except FileNotFoundError:
        return "nvcc command not found."







def run_command(command, cwd=None):
    """Run shell commands and stream output."""
    print(f"Running command: {command}")
    with subprocess.Popen(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            print(line, end='')
        for line in proc.stderr:
            print(line, end='', file=sys.stderr)
        proc.wait()
        if proc.returncode != 0:
            print(f"Command failed with exit code {proc.returncode}", file=sys.stderr)
            sys.exit(proc.returncode)


def write_requirements_file(filename, content):
    """Write requirements content to a file."""
    with open(filename, "w") as file:
        file.write(content)
    print(f"{filename} has been created.")


def delete_if_exists(path, command):
    """Delete a file or directory if it exists."""
    if os.path.exists(path):
        run_command(command)
        print(f"Deleted {path}.")


def download_file(url, destination):
    """Download a file from a URL and save it to the specified destination."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(destination, 'wb') as file:
                file.write(response.content)
            print(f'File downloaded successfully: {destination}')
        else:
            print(f'Failed to retrieve the file from {url}. Status code: {response.status_code}')
    except Exception as e:
        print(f'An error occurred: {e}')


def fix():
  url = "https://huggingface.co/NeuralFalcon/bugs/raw/main/basicsr/degradations.py"
  full_version = sys.version.split(' ')[0]
  major_minor_version = '.'.join(full_version.split('.')[:2])
  basicsr_path=f"/usr/local/lib/python{major_minor_version}/dist-packages/basicsr/data/degradations.py"
  download_file(url, basicsr_path)


# Define requirements content
requirements_content = """pytorch-lightning==2.3.3
torchmetrics==0.7.0
# torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
scipy==1.13.1
numpy==1.26.4
tqdm==4.66.4
espnet==202301
moviepy==1.0.3
python_speech_features==0.6
facexlib
tb-nightly -i https://mirrors.aliyun.com/pypi/simple
gfpgan
pydub
edge-tts==6.1.10
gradio==4.19.1
deep_translator==1.11.4
soundfile==0.10.3.post1
transformers==4.27.0
scenedetect==0.6.4
librosa
auto-editor
"""

# Write requirements to file


# Check and handle existing directories or virtual environments
if os.path.exists("AniTalker"):
    reinstall = input("AniTalker Folder already exists. Do you want to reinstall? (y/n): ").strip().lower()
    if reinstall == 'y':
        delete_if_exists("AniTalker", "rmdir /S /Q AniTalker" if os.name == "nt" else "rm -rf AniTalker")
        # delete_if_exists("myenv", "rmdir /S /Q myenv" if os.name == "nt" else "rm -rf myenv")
    else:
        print("Skipping reinstallation.")
        sys.exit(0)

# Install Git LFS and clone repositories
run_command("git lfs install")
run_command("git clone https://github.com/X-LANCE/AniTalker.git")
write_requirements_file("AniTalker/local_requirements.txt", requirements_content)
os.chdir("AniTalker")
run_command("git clone https://huggingface.co/taocode/anitalker_ckpts ckpts")
os.chdir("..")
pip_torch_command = torch_command()
print(pip_torch_command)
os.chdir("AniTalker")
if 'COLAB_GPU' in os.environ or 'KAGGLE_URL_BASE' in os.environ:
    # Get the CUDA version
    run_command(pip_torch_command)
    run_command("pip install -r local_requirements.txt")
else:
    # Create and set up virtual environment
    run_command("python -m venv myenv")
    pip_command = f"myenv\\Scripts\\{pip_torch_command}" if os.name == "nt" else "myenv/bin/{pip_torch_command}}"
    run_command(pip_command)
    pip_command = "myenv\\Scripts\\pip install -r local_requirements.txt" if os.name == "nt" else "myenv/bin/pip install -r local_requirements.txt"
    run_command(pip_command)
os.chdir("..")
# Download additional files
download_file('https://huggingface.co/NeuralFalcon/bugs/raw/main/AniTalker/extract_audio_features.py', 'AniTalker/code/extract_audio_features.py')
if 'COLAB_GPU' in os.environ or 'KAGGLE_URL_BASE' in os.environ:
  fix()
else:
  download_file('https://huggingface.co/NeuralFalcon/bugs/raw/main/basicsr/degradations.py', 'AniTalker/myenv/Lib/site-packages/basicsr/data/degradations.py')

print("Setup complete.")
