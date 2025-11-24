# ASPIICS WEB

Streamlit-based web application for the calibration and analysis of ASPIICS data.

## Installation:

### 1. Install Git and Git LFS

Git LFS is required for the resources folder (large files).

### Linux (Ubuntu/Debian)

```bash
sudo apt install git git-lfs
```

### macOS 
Using [Homebrew](https://brew.sh/):
```bash
brew install git git-lfs
```

### Windows
Download and install from the official sites:
- [Git and Git-LFS](https://git-scm.com/install/windows)


Alternatively, if you use Anaconda (any OS):

```bash
conda install -c conda-forge git git-lfs
```

### 2. Clone the repository

```bash
git clone https://gitlab.com/riccardo.chiartano/aspiics-web.git
```
### 3. Install python dependencies

```bash
pip install -r requirements.txt
```

If some libraries cause issues, consider using a virtual environment:

```bash
python -m venv /path/virt_env_name
source virt_env_name/bin/activate
pip install -r requirements.txt
```

## Launch the application:

```bash
streamlit run app.py
```
