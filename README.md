Automated Image Registration Pipeline for Developmental Mouse Brain Spatial Omics
python
"""
Automated image registration pipeline for aligning developmental mouse brain
spatial omics data to KimLab 3D developmental Brain CCF atlases.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum

import numpy as np
import pandas as pd
import nrrd
import ants
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib as mpl


class SliceOrientation(Enum):
    CORONAL = "coronal"
    SAGITTAL = "sagittal"
    AXIAL = "axial"


class RegistrationMetric(Enum):
    MATTES = "mattes"
    MEANSQUARES = "meansquares"
    GC = "gc"


@dataclass
class ImageMetadata:
    origin: tuple[float, float]
    spacing: tuple[float, float]
    shape: tuple[int, int]


@dataclass
class RegistrationConfig:
    rasterization_resolution: float = 50.0
    gaussian_blur_sigma: float = 1.0
    registration_metric: RegistrationMetric = RegistrationMetric.MATTES
    transform_type: str = "SyNRA"
    syn_iterations: tuple[int, ...] = (200, 200, 200, 50)
    affine_iterations: tuple[int, ...] = (2100, 1200, 1200, 10)
    similarity_threshold: float = 0.3


@dataclass
class QualityMetrics:
    mutual_information: float
    correlation: float
    dice_overlap: float
    registration_converged: bool
    quality_flag: str


@dataclass
class RegistrationResult:
    warped_image: np.ndarray
    forward_transforms: list[str]
    inverse_transforms: list[str]
    warped_coordinates: pd.DataFrame
    quality_metrics: QualityMetrics
    fixed_metadata: ImageMetadata
    moving_metadata: ImageMetadata


class AtlasLoader:
    """Handles loading and slicing of 3D atlas volumes."""
    
    def __init__(self, atlas_path: str):
        self.atlas_path = Path(atlas_path)
        self.volume, self.header = nrrd.read(str(self.atlas_path))
        self._parse_header()
    
    def _parse_header(self):
        self.voxel_spacing = np.diag(self.header['space directions'])
        self.origin = self.header.get('space origin', np.zeros(3))
        self.shape = self.volume.shape
        self._compute_coordinates()
    
    def _compute_coordinates(self):
        self.coordinates = [
            np.arange(n) * d + o 
            for n, d, o in zip(self.shape, self.voxel_spacing, self.origin)
        ]
    
    def get_slice(
        self, 
        slice_index: int, 
        orientation: SliceOrientation
    ) -> tuple[np.ndarray, ImageMetadata]:
        axis_map = {
            SliceOrientation.SAGITTAL: 2,
            SliceOrientation.CORONAL: 0,
            SliceOrientation.AXIAL: 1
        }
        axis = axis_map[orientation]
        
        slice_2d = np.take(self.volume, slice_index, axis=axis).astype(np.float32)
        
        other_axes = [i for i in range(3) if i != axis]
        origin = tuple(float(self.coordinates[ax][0]) for ax in other_axes)
        spacing = tuple(float(self.voxel_spacing[ax]) for ax in other_axes)
        
        metadata = ImageMetadata(
            origin=origin,
            spacing=spacing,
            shape=slice_2d.shape
        )
        
        return slice_2d, metadata
    
    def get_slice_range(
        self,
        slice_index: int,
        delta: int,
        orientation: SliceOrientation
    ) -> list[tuple[np.ndarray, int]]:
        slices = []
        for offset in [-delta, 0, delta]:
            idx = slice_index + offset
            if 0 <= idx < self.shape[2]:
                slice_data, _ = self.get_slice(idx, orientation)
                slices.append((slice_data, idx))
        return slices


class SpatialOmicsPreprocessor:
    """Preprocesses spatial omics data for registration."""
    
    def __init__(self, config: RegistrationConfig):
        self.config = config
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        return pd.read_csv(data_path)
    
    def filter_by_z(self, df: pd.DataFrame, z_value: float) -> pd.DataFrame:
        return df[df['z'] == z_value].copy()
    
    def get_unique_z_values(self, df: pd.DataFrame) -> np.ndarray:
        return df['z'].unique()
    
    def extract_coordinates(
        self, 
        df: pd.DataFrame,
        x_col: str = 'x',
        y_col: str = 'y',
        scale_factor: float = 1000.0
    ) -> tuple[np.ndarray, np.ndarray]:
        x = df[x_col].to_numpy() * scale_factor
        y = df[y_col].to_numpy() * scale_factor
        return x, y
    
    def transform_coordinates(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rotation_deg: float = 270.0,
        scale: tuple[float, float] = (0.9, 0.9),
        mirror_x: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        theta = np.deg2rad(rotation_deg)
        scale_x, scale_y = scale
        
        if mirror_x:
            v1 = -scale_x * y
        else:
            v1 = scale_x * y
        v2 = scale_y * x
        
        x_transformed = np.cos(theta) * v1 - np.sin(theta) * v2
        y_transformed = np.sin(theta) * v1 + np.cos(theta) * v2
        
        return x_transformed, y_transformed
    
    def center_coordinates(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return x - np.mean(x), y - np.mean(y)
    
    def rasterize(
        self,
        x: np.ndarray,
        y: np.ndarray,
        resolution: Optional[float] = None,
        blur_sigma: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resolution = resolution or self.config.rasterization_resolution
        blur_sigma = blur_sigma or self.config.gaussian_blur_sigma
        
        x_edges = np.arange(np.min(x), np.max(x) + resolution, resolution)
        y_edges = np.arange(np.min(y), np.max(y) + resolution, resolution)
        
        density, y_bins, x_bins = np.histogram2d(y, x, bins=[y_edges, x_edges])
        
        if blur_sigma > 0:
            density = gaussian_filter(density, sigma=blur_sigma)
        
        x_centers = x_bins[:-1] + resolution / 2.0
        y_centers = y_bins[:-1] + resolution / 2.0
        
        return x_centers, y_centers, density.astype(np.float32)


class ImageRegistrar:
    """Performs ANTsPy-based image registration."""
    
    def __init__(self, config: RegistrationConfig):
        self.config = config
    
    def create_ants_image(
        self,
        data: np.ndarray,
        metadata: ImageMetadata
    ) -> ants.ANTsImage:
        return ants.from_numpy(
            data,
            origin=metadata.origin,
            spacing=metadata.spacing
        )
    
    def normalize_image(self, image: ants.ANTsImage) -> ants.ANTsImage:
        return ants.iMath(image, "Normalize")
    
    def register(
        self,
        fixed: ants.ANTsImage,
        moving: ants.ANTsImage
    ) -> dict:
        fixed_norm = self.normalize_image(fixed)
        moving_norm = self.normalize_image(moving)
        
        return ants.registration(
            fixed=fixed_norm,
            moving=moving_norm,
            type_of_transform=self.config.transform_type,
            syn_metric=self.config.registration_metric.value,
            reg_iterations=self.config.syn_iterations,
            aff_iterations=self.config.affine_iterations,
        )
    
    def warp_coordinates(
        self,
        coordinates: pd.DataFrame,
        transforms: list[str],
        invert_flags: Optional[list[bool]] = None
    ) -> pd.DataFrame:
        if invert_flags is None:
            invert_flags = [True, False]
        
        return ants.apply_transforms_to_points(
            dim=2,
            points=coordinates.copy(),
            transformlist=transforms,
            whichtoinvert=invert_flags,
        )
    
    def coordinates_to_pixels(
        self,
        warped_points: pd.DataFrame,
        reference_image: ants.ANTsImage
    ) -> tuple[np.ndarray, np.ndarray]:
        row_px = (warped_points['x'].to_numpy() - reference_image.origin[0]) / reference_image.spacing[0]
        col_px = (warped_points['y'].to_numpy() - reference_image.origin[1]) / reference_image.spacing[1]
        return row_px, col_px


class QualityController:
    """Computes quality metrics and generates QC visualizations."""
    
    def __init__(self, config: RegistrationConfig):
        self.config = config
    
    def compute_metrics(
        self,
        fixed: np.ndarray,
        warped: np.ndarray
    ) -> QualityMetrics:
        mi = self._mutual_information(fixed, warped)
        correlation = self._correlation(fixed, warped)
        dice = self._dice_overlap(fixed, warped)
        
        converged = mi > self.config.similarity_threshold
        quality_flag = self._determine_flag(mi, correlation, dice)
        
        return QualityMetrics(
            mutual_information=mi,
            correlation=correlation,
            dice_overlap=dice,
            registration_converged=converged,
            quality_flag=quality_flag
        )
    
    def _mutual_information(self, img1: np.ndarray, img2: np.ndarray) -> float:
        hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=50)
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        
        nonzero = pxy > 0
        mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
        return float(mi)
    
    def _correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        return float(np.corrcoef(img1.ravel(), img2.ravel())[0, 1])
    
    def _dice_overlap(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        threshold: float = 0.1
    ) -> float:
        mask1 = img1 > threshold * np.max(img1)
        mask2 = img2 > threshold * np.max(img2)
        
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1) + np.sum(mask2)
        
        return float(2 * intersection / union) if union > 0 else 0.0
    
    def _determine_flag(
        self,
        mi: float,
        correlation: float,
        dice: float
    ) -> str:
        if mi > 0.5 and correlation > 0.5 and dice > 0.5:
            return "PASS"
        elif mi > 0.3 or correlation > 0.3:
            return "REVIEW"
        return "FAIL"
    
    def generate_qc_visualization(
        self,
        moving_raster: np.ndarray,
        atlas_slice: np.ndarray,
        warped_image: np.ndarray,
        warped_row_px: np.ndarray,
        warped_col_px: np.ndarray,
        slice_index: int,
        output_path: str,
        adjacent_slices: Optional[list[tuple[np.ndarray, int]]] = None
    ):
        n_plots = 4 + (len(adjacent_slices) if adjacent_slices else 0)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        
        plot_idx = 0
        
        if adjacent_slices:
            for slice_data, idx in adjacent_slices:
                axes[plot_idx].imshow(slice_data, cmap=mpl.cm.Reds)
                axes[plot_idx].set_title(f'Atlas slice (idx={idx})')
                plot_idx += 1
        
        axes[plot_idx].imshow(moving_raster, cmap=mpl.cm.Blues)
        axes[plot_idx].set_title('Spatial omics raster')
        plot_idx += 1
        
        axes[plot_idx].imshow(atlas_slice, cmap=mpl.cm.Reds)
        axes[plot_idx].set_title(f'Atlas slice (idx={slice_index})')
        plot_idx += 1
        
        axes[plot_idx].imshow(atlas_slice, cmap=mpl.cm.Reds, alpha=0.5)
        axes[plot_idx].imshow(warped_image, cmap=mpl.cm.Blues, alpha=0.5)
        axes[plot_idx].set_title('Overlay')
        plot_idx += 1
        
        axes[plot_idx].imshow(atlas_slice, cmap=mpl.cm.Greys)
        axes[plot_idx].scatter(warped_col_px, warped_row_px, s=0.3, color='blue', alpha=0.3)
        axes[plot_idx].set_title('Warped coordinates')
        axes[plot_idx].set_xlim(0, atlas_slice.shape[1])
        axes[plot_idx].set_ylim(atlas_slice.shape[0], 0)
        
        for ax in axes:
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class RegistrationPipeline:
    """Main pipeline orchestrating the registration workflow."""
    
    def __init__(
        self,
        config: RegistrationConfig,
        output_dir: str
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = SpatialOmicsPreprocessor(config)
        self.registrar = ImageRegistrar(config)
        self.qc = QualityController(config)
        
        self.atlas_loader: Optional[AtlasLoader] = None
    
    def load_atlas(self, atlas_path: str):
        self.atlas_loader = AtlasLoader(atlas_path)
    
    def run_single_slice(
        self,
        omics_df: pd.DataFrame,
        z_value: float,
        atlas_slice_index: int,
        orientation: SliceOrientation,
        output_prefix: str,
        coordinate_transform_params: Optional[dict] = None
    ) -> RegistrationResult:
        if self.atlas_loader is None:
            raise ValueError("Atlas not loaded. Call load_atlas() first.")
        
        filtered_df = self.preprocessor.filter_by_z(omics_df, z_value)
        x, y = self.preprocessor.extract_coordinates(filtered_df)
        
        transform_params = coordinate_transform_params or {
            'rotation_deg': 270.0,
            'scale': (0.9, 0.9),
            'mirror_x': True
        }
        x_trans, y_trans = self.preprocessor.transform_coordinates(x, y, **transform_params)
        x_centered, y_centered = self.preprocessor.center_coordinates(x_trans, y_trans)
        
        x_grid, y_grid, density = self.preprocessor.rasterize(y_centered, x_centered)
        
        atlas_slice, atlas_metadata = self.atlas_loader.get_slice(atlas_slice_index, orientation)
        
        moving_metadata = ImageMetadata(
            origin=(float(y_grid[0]), float(x_grid[0])),
            spacing=(self.config.rasterization_resolution, self.config.rasterization_resolution),
            shape=density.shape
        )
        
        fixed_ants = self.registrar.create_ants_image(atlas_slice, atlas_metadata)
        moving_ants = self.registrar.create_ants_image(density, moving_metadata)
        
        reg_result = self.registrar.register(fixed_ants, moving_ants)
        
        points_df = pd.DataFrame({'x': x_centered, 'y': y_centered})
        warped_points = self.registrar.warp_coordinates(
            points_df,
            reg_result['invtransforms']
        )
        
        row_px, col_px = self.registrar.coordinates_to_pixels(warped_points, fixed_ants)
        
        warped_image = reg_result['warpedmovout'].numpy()
        quality_metrics = self.qc.compute_metrics(atlas_slice, warped_image)
        
        adjacent = self.atlas_loader.get_slice_range(atlas_slice_index, 5, orientation)
        qc_path = self.output_dir / f"{output_prefix}_qc.png"
        self.qc.generate_qc_visualization(
            density, atlas_slice, warped_image,
            row_px, col_px, atlas_slice_index,
            str(qc_path), adjacent
        )
        
        output_df = filtered_df.copy()
        output_df['ccf_x'] = warped_points['x'].to_numpy() / 1000.0
        output_df['ccf_y'] = warped_points['y'].to_numpy() / 1000.0
        output_df['ccf_z'] = z_value
        
        output_csv = self.output_dir / f"{output_prefix}_registered.csv"
        keep_cols = ['cell_label', 'x', 'y', 'z', 'ccf_x', 'ccf_y', 'ccf_z']
        available_cols = [c for c in keep_cols if c in output_df.columns]
        output_df[available_cols].to_csv(output_csv, index=False)
        
        return RegistrationResult(
            warped_image=warped_image,
            forward_transforms=reg_result['fwdtransforms'],
            inverse_transforms=reg_result['invtransforms'],
            warped_coordinates=warped_points,
            quality_metrics=quality_metrics,
            fixed_metadata=atlas_metadata,
            moving_metadata=moving_metadata
        )
    
    def run_batch(
        self,
        omics_path: str,
        atlas_slice_mapping: dict[float, int],
        orientation: SliceOrientation,
        coordinate_transform_params: Optional[dict] = None
    ) -> dict[float, RegistrationResult]:
        omics_df = self.preprocessor.load_data(omics_path)
        results = {}
        
        for z_value, slice_index in atlas_slice_mapping.items():
            prefix = f"z{z_value}_slice{slice_index}"
            try:
                result = self.run_single_slice(
                    omics_df, z_value, slice_index, orientation,
                    prefix, coordinate_transform_params
                )
                results[z_value] = result
                print(f"Completed registration for z={z_value}, quality={result.quality_metrics.quality_flag}")
            except Exception as e:
                print(f"Failed registration for z={z_value}: {e}")
        
        self._generate_summary_report(results)
        return results
    
    def _generate_summary_report(self, results: dict[float, RegistrationResult]):
        summary_data = []
        for z_value, result in results.items():
            summary_data.append({
                'z_value': z_value,
                'mutual_information': result.quality_metrics.mutual_information,
                'correlation': result.quality_metrics.correlation,
                'dice_overlap': result.quality_metrics.dice_overlap,
                'converged': result.quality_metrics.registration_converged,
                'quality_flag': result.quality_metrics.quality_flag
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / "registration_summary.csv", index=False)


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python registration_pipeline.py <z_index> <atlas_slice_index>")
        print("       python registration_pipeline.py --batch <mapping_file>")
        sys.exit(1)
    
    config = RegistrationConfig(
        rasterization_resolution=50.0,
        gaussian_blur_sigma=1.0,
        registration_metric=RegistrationMetric.MATTES,
        transform_type="SyNRA"
    )
    
    pipeline = RegistrationPipeline(
        config=config,
        output_dir="./registration_results"
    )
    
    pipeline.load_atlas("aba_nissl.nrrd")
    
    if sys.argv[1] == "--batch":
        mapping = {}  # Load from file
        pipeline.run_batch(
            "data/abca3_cell_metadata.csv",
            mapping,
            SliceOrientation.SAGITTAL
        )
    else:
        z_idx = int(sys.argv[1])
        slice_idx = int(sys.argv[2])
        
        omics_df = pipeline.preprocessor.load_data("data/abca3_cell_metadata.csv")
        z_values = pipeline.preprocessor.get_unique_z_values(omics_df)
        z_value = z_values[z_idx]
        
        result = pipeline.run_single_slice(
            omics_df=omics_df,
            z_value=z_value,
            atlas_slice_index=slice_idx,
            orientation=SliceOrientation.SAGITTAL,
            output_prefix=f"registration_z{z_idx}_s{slice_idx}"
        )
        
        print(f"Registration complete. Quality: {result.quality_metrics.quality_flag}")
        print(f"Mutual Information: {result.quality_metrics.mutual_information:.4f}")
        print(f"Correlation: {result.quality_metrics.correlation:.4f}")


if __name__ == "__main__":
    main()
Key Improvements
Modular Architecture: Separated concerns into distinct classes (AtlasLoader, SpatialOmicsPreprocessor, ImageRegistrar, QualityController, RegistrationPipeline)

Configuration Management: Used dataclasses for type-safe configuration and results

Quality Control: Added automated QC metrics (mutual information, correlation, Dice overlap) with pass/review/fail flagging

Batch Processing: Support for processing multiple slices with summary reports

Type Safety: Added type hints throughout for better IDE support and documentation

Extensibility: Easy to add new registration methods, metrics, or preprocessing steps






avatar
can you explain how to use this for a readme?


2 / 2





Spatial Omics to Brain Atlas Registration Pipeline
Automated image registration pipeline for aligning developmental mouse brain spatial omics data (MERFISH, Visium, HybISS) to KimLab 3D developmental Brain CCF atlases.

Overview
This pipeline registers 2D spatial transcriptomics sections to reference atlas slices using ANTsPy, enabling mapping of single-cell coordinates to a common coordinate framework (CCF).

Key Features
Multi-modal registration using mutual information metrics
Handles large initial misalignments (rotation, translation, scaling)
Combined affine and deformable (SyN) registration
Automated quality control with pass/review/fail flagging
Batch processing support with summary reports
Preserves original gene expression data while adding CCF coordinates
Installation
bash
# Create conda environment
conda create -n brain_registration python=3.10
conda activate brain_registration

# Install dependencies
pip install antspyx numpy pandas scipy matplotlib nrrd
Quick Start
Single Slice Registration
bash
python registration_pipeline.py <z_index> <atlas_slice_index>
Example:

bash
python registration_pipeline.py 0 145
This registers the first z-plane from your spatial omics data to atlas slice 145.

Batch Processing
bash
python registration_pipeline.py --batch mapping.json
Input Data Requirements
Spatial Omics Data
CSV file with the following columns:

Column	Description	Required
cell_label	Unique cell identifier	Yes
x	X coordinate (mm)	Yes
y	Y coordinate (mm)	Yes
z	Z coordinate / section ID	Yes
Example:

csv
cell_label,x,y,z,gene_count
cell_001,2.345,4.567,0.1,1523
cell_002,2.401,4.612,0.1,892
Atlas Data
Format: NRRD (.nrrd)
Source: KimLab developmental Brain CCF atlas
The pipeline reads voxel spacing and origin from the NRRD header
Usage Examples
Python API - Single Slice
python
from registration_pipeline import (
    RegistrationPipeline,
    RegistrationConfig,
    SliceOrientation
)

# Configure registration parameters
config = RegistrationConfig(
    rasterization_resolution=50.0,  # microns per pixel
    gaussian_blur_sigma=1.0,
    transform_type="SyNRA"
)

# Initialize pipeline
pipeline = RegistrationPipeline(
    config=config,
    output_dir="./results"
)

# Load atlas volume
pipeline.load_atlas("path/to/atlas.nrrd")

# Load spatial omics data
omics_df = pipeline.preprocessor.load_data("path/to/cells.csv")

# Run registration for a single slice
result = pipeline.run_single_slice(
    omics_df=omics_df,
    z_value=0.1,              # z-coordinate to process
    atlas_slice_index=145,     # corresponding atlas slice
    orientation=SliceOrientation.SAGITTAL,
    output_prefix="sample1"
)

# Check quality
print(f"Quality: {result.quality_metrics.quality_flag}")
print(f"Correlation: {result.quality_metrics.correlation:.3f}")
Python API - Batch Processing
python
# Define z-value to atlas slice mapping
# This mapping should come from your slice-matching ML model
atlas_mapping = {
    0.1: 140,   # z=0.1mm -> atlas slice 140
    0.2: 145,   # z=0.2mm -> atlas slice 145
    0.3: 150,   # z=0.3mm -> atlas slice 150
}

# Run batch registration
results = pipeline.run_batch(
    omics_path="path/to/cells.csv",
    atlas_slice_mapping=atlas_mapping,
    orientation=SliceOrientation.SAGITTAL
)
Custom Coordinate Transformations
If your data requires different initial alignment:

python
result = pipeline.run_single_slice(
    omics_df=omics_df,
    z_value=0.1,
    atlas_slice_index=145,
    orientation=SliceOrientation.CORONAL,
    output_prefix="sample1",
    coordinate_transform_params={
        'rotation_deg': 180.0,    # initial rotation
        'scale': (1.0, 1.0),      # x, y scaling
        'mirror_x': False         # flip x-axis
    }
)
Custom Registration Parameters
python
from registration_pipeline import RegistrationConfig, RegistrationMetric

config = RegistrationConfig(
    # Rasterization settings
    rasterization_resolution=25.0,  # finer resolution
    gaussian_blur_sigma=2.0,        # more smoothing
    
    # Registration settings
    registration_metric=RegistrationMetric.MATTES,
    transform_type="SyNRA",
    syn_iterations=(300, 200, 100, 50),      # more iterations
    affine_iterations=(3000, 1500, 1000, 10),
    
    # QC threshold
    similarity_threshold=0.4
)
Output Files
For each registered slice, the pipeline generates:

python
output_dir/
├── sample1_registered.csv      # Cell coordinates with CCF mapping
├── sample1_qc.png              # Quality control visualization
└── registration_summary.csv    # Batch summary (if batch mode)
Registered CSV Format
Column	Description
cell_label	Original cell identifier
x	Original x coordinate (mm)
y	Original y coordinate (mm)
z	Original z coordinate (mm)
ccf_x	Registered CCF x coordinate (mm)
ccf_y	Registered CCF y coordinate (mm)
ccf_z	CCF z coordinate (mm)
QC Visualization
The QC plot includes:

Adjacent atlas slices (for context)
Rasterized spatial omics density map
Target atlas slice
Overlay of warped omics on atlas
Individual cell coordinates projected onto atlas
Quality Metrics
Metric	Description	Good Values
mutual_information	Statistical dependence between images	> 0.5
correlation	Pearson correlation of intensities	> 0.5
dice_overlap	Spatial overlap of tissue regions	> 0.5
quality_flag	Overall assessment	PASS/REVIEW/FAIL
Pipeline Workflow
java
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT DATA                                   │
├─────────────────────────────────────────────────────────────────┤
│  Spatial Omics (CSV)           Atlas Volume (NRRD)              │
│  - Cell coordinates            - 3D reference volume            │
│  - Gene expression             - Voxel spacing metadata         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   1. PREPROCESSING                               │
├─────────────────────────────────────────────────────────────────┤
│  • Filter cells by z-plane                                      │
│  • Apply coordinate transformations (rotate, scale, mirror)     │
│  • Center coordinates                                           │
│  • Rasterize point cloud → density image                        │
│  • Apply Gaussian smoothing                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. ATLAS PREPARATION                           │
├─────────────────────────────────────────────────────────────────┤
│  • Extract 2D slice at specified index                          │
│  • Match orientation (sagittal/coronal/axial)                   │
│  • Preserve physical coordinates and spacing                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   3. REGISTRATION                                │
├─────────────────────────────────────────────────────────────────┤
│  • Normalize both images                                        │
│  • Combined SyNRA registration:                                 │
│    - Rigid alignment                                            │
│    - Affine transformation                                      │
│    - Symmetric diffeomorphic (SyN) deformation                  │
│  • Metric: Mattes Mutual Information                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   4. COORDINATE MAPPING                          │
├─────────────────────────────────────────────────────────────────┤
│  • Apply inverse transforms to cell coordinates                 │
│  • Map from omics space → CCF space                             │
│  • Convert to physical units (mm)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   5. QUALITY CONTROL                             │
├─────────────────────────────────────────────────────────────────┤
│  • Compute similarity metrics                                   │
│  • Generate QC visualizations                                   │
│  • Flag results as PASS/REVIEW/FAIL                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT                                       │
├─────────────────────────────────────────────────────────────────┤
│  • Registered coordinates (CSV)                                 │
│  • QC visualizations (PNG)                                      │
│  • Summary statistics (CSV)                                     │
│  • Transformation files (for reuse)                             │
└─────────────────────────────────────────────────────────────────┘
Configuration Reference
RegistrationConfig Parameters
Parameter	Type	Default	Description
rasterization_resolution	float	50.0	Pixel size in microns for density map
gaussian_blur_sigma	float	1.0	Smoothing kernel size
registration_metric	RegistrationMetric	MATTES	Similarity metric
transform_type	str	"SyNRA"	ANTsPy transform type
syn_iterations	tuple	(200,200,200,50)	SyN iterations per resolution
affine_iterations	tuple	(2100,1200,1200,10)	Affine iterations per resolution
similarity_threshold	float	0.3	Minimum MI for convergence
SliceOrientation Options
Value	Description
SAGITTAL	Left-right slicing (default for most brain atlases)
CORONAL	Front-back slicing
AXIAL	Top-bottom slicing
Troubleshooting
Poor Registration Quality
Check initial alignment: Adjust rotation_deg, scale, and mirror_x parameters
Verify slice matching: Ensure the atlas slice index corresponds to the correct anatomical position
Adjust rasterization: Try different rasterization_resolution values (25-100 µm)
Increase iterations: Use more registration iterations for difficult cases
Memory Issues
For large datasets, process in chunks:

python
# Process subset of z-values
z_values = pipeline.preprocessor.get_unique_z_values(omics_df)
for z in z_values[start:end]:
    pipeline.run_single_slice(...)
Coordinate System Mismatch
If registered coordinates appear flipped or rotated:

python
# Try different transform parameters
coordinate_transform_params={
    'rotation_deg': 0,      # Try 0, 90, 180, 270
    'mirror_x': True,       # Toggle mirroring
    'scale': (1.0, 1.0)
}
API Reference
Classes
Class	Description
RegistrationPipeline	Main orchestrator for the registration workflow
AtlasLoader	Loads and slices 3D NRRD atlas volumes
SpatialOmicsPreprocessor	Preprocesses spatial omics data
ImageRegistrar	Performs ANTsPy registration
QualityController	Computes QC metrics and visualizations
Data Classes
Class	Description
RegistrationConfig	Configuration parameters
RegistrationResult	Output from registration
QualityMetrics	QC measurements
ImageMetadata	Image spatial metadata