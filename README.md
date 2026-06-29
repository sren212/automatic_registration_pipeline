# Developmental Mouse Brain Image Registration Pipeline

An automated Python-based image registration pipeline designed to align 2D developmental mouse brain spatial omics data with the KimLab 3D Developmental Brain Common Coordinate Framework (CCF) atlases. 

The pipeline handles coordinate pre-processing (scaling, rotation, mirroring), rasterizes discrete cell coordinates into 2D density maps, performs robust rigid-to-deformable registrations using `ANTsPy` (Symmetric Normalization - `SyNRA`), transforms spatial data into atlas space, and outputs comprehensive quality control (QC) metrics and visualizations.

---

## Features

* **Flexible Slice Extraction**: Supports extraction of coronal, sagittal, and axial reference slices directly from 3D NRRD atlas volumes.
* **Spatial Data Pre-processing**: Built-in coordinate conversion, centering, rigid transformations, and configurable 2D density rasterization with Gaussian smoothing.
* **Advanced Registration Engine**: Powered by ANTs Py (Advanced Normalization Tools), supporting various optimization metrics (`Mattes Mutual Information`, `Mean Squares`, `Cross-Correlation`) and registration routines (`SyNRA`).
* **Coordinate Warping**: Automatically applies calculated inverse deformation fields back to discrete single-cell point coordinates.
* **Automated Quality Control**: Evaluates registration performance using Mutual Information, Pearson Correlation Coefficient, and Dice Overlap with a deterministic `PASS`/`REVIEW`/`FAIL` grading scheme.
* **Rich Diagnostic Visualizations**: Outputs 4-panel diagnostic plots capturing the raw density data, reference atlas slice, structural overlay, and final warped coordinates.

---

## Installation

Ensure you have the required system and Python dependencies installed.

```bash
pip install numpy pandas pynrrd antspyx scipy matplotlib
```

---

## Example Run

```bash
ce_13_cell_metadata_rot30.csv --slice 64
```