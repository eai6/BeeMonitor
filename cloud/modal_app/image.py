"""Modal container image definition for BeeMonitor Cloud.

Builds a Debian-based image with all Python dependencies and the
beemonitor package pre-installed.
"""

import modal

beemonitor_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install(
        # Core ML / CV
        "ultralytics>=8.4.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python-headless>=4.8.0",
        # ML
        "scikit-learn>=1.3.0",
        "filterpy>=1.4.5",
        "scipy>=1.11.0",
        "joblib>=1.3.0",
        # Data
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        # Azure
        "azure-storage-blob>=12.19.0",
        "azure-identity>=1.15.0",
        # Utils
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
    )
    .add_local_dir("src/beemonitor", "/root/src/beemonitor")
    .add_local_dir("cloud", "/root/cloud")
    .run_commands(
        "echo 'import sys; sys.path.insert(0, \"/root/src\"); sys.path.insert(0, \"/root\")' "
        "> /root/sitecustomize.py",
        "cp /root/sitecustomize.py /usr/lib/python3.12/sitecustomize.py",
    )
)
