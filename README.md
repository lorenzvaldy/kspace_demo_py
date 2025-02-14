# KSpace_Demo_Py

[![Conda](https://img.shields.io/conda/dn/conda-forge/python)](https://docs.conda.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of the K-Space MRI simulation originally developed in MATLAB. This project demonstrates k-space fundamentals through interactive visualization of MRI acquisition and reconstruction.

![Demo Screenshot](imgs/demo_interface.png)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/lorenzvaldy/kspace_demo_py.git
cd kspace_demo_py

conda create -n kspace-demo python=3.9
conda activate kspace-demo

pip install -r requirements.txt
```

### 2. Run the Demo
```bash
python kspace_demo.py
```

## Usage

1. Select input image (built-in phantoms or custom file)

2. Configure acquisition parameters:
    * **Sequence type**: Choose between `spiral` or `EPI`
    * **Field of View**: Adjust imaging area
    * **Resolution**: Set image resolution
    * **Noise characteristics**: Configure noise parameters
        * Local offset
        * Random offset
        * Random lowpass
        * X gradient
        * Y gradient
        * DC offset
        * Map
        * None

3. View real-time reconstruction / Direct Reconstruction

4. Compare original vs reconstructed images

## Attribution

This project ports the MATLAB implementation from:
"kspace_demo" by WinawerLab
https://github.com/WinawerLab/kspace_demo.git

## License 📄

MIT License - see [LICENSE](LICENSE) for details