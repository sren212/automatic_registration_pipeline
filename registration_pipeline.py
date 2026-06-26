"""
Automated image registration pipeline for aligning developmental mouse brain
spatial omics data to KimLab 3D developmental Brain CCF atlases.
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
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


class SpatialOmicsPreprocessor:
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
        output_path: str
    ):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(moving_raster, cmap=mpl.cm.Blues)
        axes[0].set_title('Moving image raster')

        axes[1].imshow(atlas_slice, cmap=mpl.cm.Reds)
        axes[1].set_title(f'Atlas slice (idx={slice_index})')

        axes[2].imshow(atlas_slice, cmap=mpl.cm.Reds, alpha=0.5)
        axes[2].imshow(warped_image, cmap=mpl.cm.Blues, alpha=0.5)
        axes[2].set_title('Overlay')

        axes[3].imshow(atlas_slice, cmap=mpl.cm.Greys)
        axes[3].scatter(warped_col_px, warped_row_px, s=0.3, color='blue', alpha=0.3)
        axes[3].set_title('Warped coordinates')
        axes[3].set_xlim(0, atlas_slice.shape[1])
        axes[3].set_ylim(atlas_slice.shape[0], 0)

        for ax in axes:
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class RegistrationPipeline:
    def __init__(self, config: RegistrationConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = SpatialOmicsPreprocessor(config)
        self.registrar = ImageRegistrar(config)
        self.qc = QualityController(config)
        self.atlas_loader: Optional[AtlasLoader] = None

    def load_atlas(self, atlas_path: str):
        self.atlas_loader = AtlasLoader(atlas_path)
        print(f"Loaded atlas: {atlas_path}")
        print(f"  Shape: {self.atlas_loader.shape}")
        print(f"  Voxel spacing: {self.atlas_loader.voxel_spacing}")

    def run(
        self,
        omics_path: str,
        atlas_slice_index: int,
        orientation: SliceOrientation,
        output_prefix: Optional[str] = None,
        coordinate_transform_params: Optional[dict] = None
    ) -> RegistrationResult:
        if self.atlas_loader is None:
            raise ValueError("Atlas not loaded. Call load_atlas() first.")

        if output_prefix is None:
            output_prefix = Path(omics_path).stem + f"_to_slice{atlas_slice_index}"

        omics_df = self.preprocessor.load_data(omics_path)
        z_values = self.preprocessor.get_unique_z_values(omics_df)

        if len(z_values) > 1:
            print(f"Warning: Multiple z-values found: {z_values}. Using first value.")

        z_value = z_values[0]
        filtered_df = self.preprocessor.filter_by_z(omics_df, z_value)
        print(f"Processing {len(filtered_df)} cells at z={z_value}")

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

        print("Running registration...")
        reg_result = self.registrar.register(fixed_ants, moving_ants)

        points_df = pd.DataFrame({'x': x_centered, 'y': y_centered})
        warped_points = self.registrar.warp_coordinates(
            points_df,
            reg_result['invtransforms']
        )

        row_px, col_px = self.registrar.coordinates_to_pixels(warped_points, fixed_ants)

        warped_image = reg_result['warpedmovout'].numpy()
        quality_metrics = self.qc.compute_metrics(atlas_slice, warped_image)

        qc_path = self.output_dir / f"{output_prefix}_qc.png"
        self.qc.generate_qc_visualization(
            density, atlas_slice, warped_image,
            row_px, col_px, atlas_slice_index,
            str(qc_path)
        )
        print(f"Saved QC visualization: {qc_path}")

        output_df = filtered_df.copy()
        output_df['ccf_x'] = warped_points['x'].to_numpy() / 1000.0
        output_df['ccf_y'] = warped_points['y'].to_numpy() / 1000.0
        output_df['ccf_z'] = z_value

        output_csv = self.output_dir / f"{output_prefix}_registered.csv"
        keep_cols = ['cell_label', 'x', 'y', 'z', 'ccf_x', 'ccf_y', 'ccf_z']
        available_cols = [c for c in keep_cols if c in output_df.columns]
        output_df[available_cols].to_csv(output_csv, index=False)
        print(f"Saved registered coordinates: {output_csv}")

        return RegistrationResult(
            warped_image=warped_image,
            forward_transforms=reg_result['fwdtransforms'],
            inverse_transforms=reg_result['invtransforms'],
            warped_coordinates=warped_points,
            quality_metrics=quality_metrics,
            fixed_metadata=atlas_metadata,
            moving_metadata=moving_metadata
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register spatial omics data to brain atlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python registration_pipeline.py \\
      --atlas data/aba_nissl.nrrd \\
      --omics data/abca3_slice_13_cell_metadata.csv \\
      --slice 64

  python registration_pipeline.py \\
      -a data/aba_nissl.nrrd \\
      -i data/cells.csv \\
      -s 64 \\
      --rotation 180 \\
      --scale 1.0 1.0 \\
      --no-mirror
        """
    )

    parser.add_argument("--atlas", "-a", required=True, help="Path to atlas NRRD file")
    parser.add_argument("--omics", "-i", required=True, help="Path to spatial omics CSV file")
    parser.add_argument("--slice", "-s", type=int, required=True, help="Atlas slice index")
    parser.add_argument(
        "--orientation", "-r",
        choices=["sagittal", "coronal", "axial"],
        default="sagittal",
        help="Slice orientation (default: sagittal)"
    )
    parser.add_argument("--output", "-o", default="./registration_results", help="Output directory")
    parser.add_argument("--prefix", "-p", default=None, help="Output file prefix")
    parser.add_argument("--rotation", type=float, default=270.0, help="Initial rotation in degrees")
    parser.add_argument(
        "--scale",
        type=float,
        nargs=2,
        default=[0.9, 0.9],
        metavar=("X", "Y"),
        help="Scale factors for x and y"
    )
    parser.add_argument("--no-mirror", action="store_true", help="Disable x-axis mirroring")
    parser.add_argument("--resolution", type=float, default=50.0, help="Rasterization resolution in microns")
    parser.add_argument("--blur", type=float, default=1.0, help="Gaussian blur sigma")

    return parser.parse_args()


def main():
    args = parse_args()

    orientation_map = {
        "sagittal": SliceOrientation.SAGITTAL,
        "coronal": SliceOrientation.CORONAL,
        "axial": SliceOrientation.AXIAL
    }

    config = RegistrationConfig(
        rasterization_resolution=args.resolution,
        gaussian_blur_sigma=args.blur,
        registration_metric=RegistrationMetric.MATTES,
        transform_type="SyNRA"
    )

    pipeline = RegistrationPipeline(config=config, output_dir=args.output)
    pipeline.load_atlas(args.atlas)

    coordinate_transform_params = {
        'rotation_deg': args.rotation,
        'scale': tuple(args.scale),
        'mirror_x': not args.no_mirror
    }

    print(f"\n{'='*60}")
    print("Registration Parameters")
    print(f"{'='*60}")
    print(f"Atlas:       {args.atlas}")
    print(f"Omics data:  {args.omics}")
    print(f"Atlas slice: {args.slice}")
    print(f"Orientation: {args.orientation}")
    print(f"Rotation:    {args.rotation}°")
    print(f"Scale:       {args.scale}")
    print(f"Mirror X:    {not args.no_mirror}")
    print(f"Resolution:  {args.resolution} µm")
    print(f"Blur sigma:  {args.blur}")
    print(f"Output:      {args.output}")
    print(f"{'='*60}\n")

    result = pipeline.run(
        omics_path=args.omics,
        atlas_slice_index=args.slice,
        orientation=orientation_map[args.orientation],
        output_prefix=args.prefix,
        coordinate_transform_params=coordinate_transform_params
    )

    print(f"\n{'='*60}")
    print("Registration Results")
    print(f"{'='*60}")
    print(f"Quality Flag:        {result.quality_metrics.quality_flag}")
    print(f"Mutual Information:  {result.quality_metrics.mutual_information:.4f}")
    print(f"Correlation:         {result.quality_metrics.correlation:.4f}")
    print(f"Dice Overlap:        {result.quality_metrics.dice_overlap:.4f}")
    print(f"Converged:           {result.quality_metrics.registration_converged}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()