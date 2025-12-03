# GPU-Accelerated Nearest Neighbor Search for 3D Point Cloud Registration

This repository contains the reference implementation of **VANICP**, a GPU-accelerated nearest neighbor search and efficient point cloud registration pipeline.  

---

## ✨ Features

- ⚡ High-performance CUDA implementation  
- 🔍 Voxel-based Nearest Neighbor Search 
- 🧩 Full ICP pipeline (SVD-based transform estimation)  
- 📈 Designed for large-scale point cloud registration  
- 🧠 Includes CUDA kernels for:
  - Voxelization
  - Dilation-based voxel filling
  - Nearest neighbor search
  - SVD-based transformation estimation (via cuSOLVER)
- 📁 Works out of the box with standard datasets (e.g., Stanford Bunny)

---

## 📂 Repository Structure
```
vanicp/
│── src/
│ ├── vanicp.cu # Main GPU kernels
│ ├── io.cpp/.h # File I/O utilities
│ ├── utils.cu # CUDA helpers
│ ├── main.cpp # Registration entry
│── data/
│ ├── source.txt
│ ├── target.txt
│── Eigen/
│── CMakeLists.txt
│── LICENSE
│── README.md
```


---

## 📦 Requirements

- **CUDA Toolkit 11.0+** (tested on CUDA 11/12)
- **CMake ≥ 3.18**
- **Eigen3**
- A modern NVIDIA GPU  
  (tested on RTX 4090, A6000, Jetson AGX Xavier)

---

## 🔧 Build Instructions

```bash
mkdir build
cd build
cmake ..
make -j
```
## 🚀 Running VANICP
Run: 
```bash
./vanicp ../data/ source.txt target.txt
```

## 📜 License — MIT
This project is released under the [MIT License](LICENSE).