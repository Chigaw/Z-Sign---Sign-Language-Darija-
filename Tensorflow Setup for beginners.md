# TensorFlow Environment Setup
This is a small guide for beginners to Setup tensorflow on their machines.
## Requirements
- Anaconda
  In my case I have the version with python 3.9.7, pay attention to which version you're installing because tensorflow is not compatible with all python version (at this point of time the latest python compatible is 3.11 and for tensorflow gpu the latest compatible is 3.10)
## Links
- [TensorFlow](https://www.tensorflow.org/install/source_windows#cpu)
  And for those who have an NVIDIA GPU you might also need these for a faster training:
- [TensorFlow GPU](https://www.tensorflow.org/install/source_windows#gpu)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [cuDNN](https://developer.nvidia.com/cudnn)

## Virtual Environment Setup
1. Open a terminal.
2. Create a virtual environment then activate it (with name being whatever name you want):
  #### Using `venv`:
   ```bash
   python -m venv name
   ```
   ```bash
   .\name\Scripts\activate
   ```
  #### Using `conda`:
   ```bash
   conda create -n name
   ```
   ```bash
   conda activate name
   ```
3. Install ipykernel :
   ### Using pip
   ```bash
   pip install ipykernel
   ```
   ### Using conda
   ```bash
   conda install ipykernel
   ```
4. Create kernel :
   ```bash
   python -m ipykernel install --user --name=name
   ```
5. Install requirements (only install tensorflow-gpu if you have NVIDIA) :
   
   in my case I installed the versions 2.10.0, the version you'll need depends on what python you have (check the tensorflow link above)
   ```bash
   pip install tensorflow==2.10.0 tensorflow-gpu==2.10.0 opencv-python matplotlib
   ```
7. For those who installed tensorflow-gpu:
   
   First of all check the tensorflow website for which versions you need for CUDA and cuDNN, in my case it was 11.2 and 8.1. Run these commands below
    ```bash
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.1
     ```
    ```bash
    set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
     ```
