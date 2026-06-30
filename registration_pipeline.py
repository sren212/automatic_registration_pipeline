#!/usr/bin/env python3
"""
Automated image registration pipeline for aligning developmental mouse brain
spatial omics data to KimLab 3D developmental Brain CCF atlases using ANTsPy.
"""

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import ants
import matplotlib as mpl
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


class SliceOrientation(Enum):
    CORONAL = 0
    AXIAL = 1
    SAGITTAL = 2


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
    rotation_deg: float = 0.0
    scale: tuple[float, float] = (0.9, 0.9)
    mirror_x: bool = False
    registration_metric: RegistrationMetric = RegistrationMetric.MATTES
    transform_type: str = "SyNRA"
    syn_iterations: tuple[int, ...] = (200, 200, 200, 50)
    affine_iterations: tuple[int, ...] = (2100, 1200, 1200, 10)
    similarity_threshold: float = 0.3
    coordinate_scale: float = 1000.0
    search_rotation: bool = False
    rotation_step: float = 90.0
    search_mirror: bool = False


@dataclass
class QualityMetrics:
    mutual_information: float
    correlation: float
    dice_overlap: float
    pre_registration_dice: float
    pre_registration_correlation: float
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
    def __init__(self, atlas_path: Path):
        self.atlas_path = atlas_path
        self.volume, self.header = nrrd.read(str(atlas_path))
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

    def get_slice(self, slice_index: int, orientation: SliceOrientation) -> tuple[np.ndarray, ImageMetadata]:
        axis = orientation.value
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

    def load_data(self, data_path: Path) -> pd.DataFrame:
        return pd.read_csv(data_path)

    def extract_coordinates(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        coord_cols = self._find_coordinate_columns(df)
        x = df[coord_cols[0]].to_numpy() * self.config.coordinate_scale
        y = df[coord_cols[1]].to_numpy() * self.config.coordinate_scale
        return y, x

    def _find_coordinate_columns(self, df: pd.DataFrame) -> list[str]:
        patterns = [
            ["x", "y"],
            ["X", "Y"],
            ["global_x", "global_y"],
            ["centroid_x", "centroid_y"],
        ]
        for pattern in patterns:
            if all(col in df.columns for col in pattern):
                return pattern
        raise ValueError("Could not identify coordinate columns")

    def transform_coordinates(self, x: np.ndarray, y: np.ndarray,
                              rotation_deg: Optional[float] = None,
                              mirror_x: Optional[bool] = None) -> tuple[np.ndarray, np.ndarray]:
        theta = np.deg2rad(rotation_deg if rotation_deg is not None else self.config.rotation_deg)
        scale_x, scale_y = self.config.scale
        use_mirror = mirror_x if mirror_x is not None else self.config.mirror_x

        if use_mirror:
            v1 = -scale_x * y
        else:
            v1 = scale_x * y
        v2 = scale_y * x

        ap_aligned = np.cos(theta) * v1 - np.sin(theta) * v2
        dv_aligned = np.sin(theta) * v1 + np.cos(theta) * v2

        return ap_aligned, dv_aligned

    def center_coordinates(self, ap: np.ndarray, dv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ap - np.mean(ap), dv - np.mean(dv)

    def rasterize(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resolution = self.config.rasterization_resolution
        blur_sigma = self.config.gaussian_blur_sigma

        x_edges = np.arange(np.min(x), np.max(x) + resolution, resolution)
        y_edges = np.arange(np.min(y), np.max(y) + resolution, resolution)

        density, y_bins, x_bins = np.histogram2d(y, x, bins=[y_edges, x_edges])

        if blur_sigma > 0:
            density = gaussian_filter(density, sigma=blur_sigma)

        X_grid = x_bins[:-1] + resolution / 2.0
        Y_grid = y_bins[:-1] + resolution / 2.0

        return X_grid, Y_grid, density.astype(np.float32)

    def rasterize_to_atlas_grid(self, x: np.ndarray, y: np.ndarray,
                                atlas_metadata: ImageMetadata) -> np.ndarray:
        shape = atlas_metadata.shape
        origin = atlas_metadata.origin
        spacing = atlas_metadata.spacing

        density = np.zeros(shape, dtype=np.float32)

        col_idx = ((x - origin[0]) / spacing[0]).astype(int)
        row_idx = ((y - origin[1]) / spacing[1]).astype(int)

        valid = (col_idx >= 0) & (col_idx < shape[1]) & \
                (row_idx >= 0) & (row_idx < shape[0])

        np.add.at(density, (row_idx[valid], col_idx[valid]), 1)

        if self.config.gaussian_blur_sigma > 0:
            density = gaussian_filter(density, sigma=self.config.gaussian_blur_sigma)

        return density


class ImageRegistrar:
    def __init__(self, config: RegistrationConfig):
        self.config = config

    def create_ants_image(self, data: np.ndarray, metadata: ImageMetadata) -> ants.ANTsImage:
        return ants.from_numpy(data, origin=metadata.origin, spacing=metadata.spacing)

    def normalize_image(self, image: ants.ANTsImage) -> ants.ANTsImage:
        return ants.iMath(image, "Normalize")

    def register(self, fixed: ants.ANTsImage, moving: ants.ANTsImage) -> dict:
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

    def warp_coordinates(self, coordinates: pd.DataFrame, transforms: list[str]) -> pd.DataFrame:
        return ants.apply_transforms_to_points(
            dim=2,
            points=coordinates.copy(),
            transformlist=transforms,
            whichtoinvert=[True, False],
        )

    def affine_register(self, fixed: ants.ANTsImage, moving: ants.ANTsImage) -> dict:
        fixed_norm = self.normalize_image(fixed)
        moving_norm = self.normalize_image(moving)

        return ants.registration(
            fixed=fixed_norm,
            moving=moving_norm,
            type_of_transform="Affine",
            aff_iterations=self.config.affine_iterations,
        )


class QualityController:
    def __init__(self, config: RegistrationConfig):
        self.config = config

    def compute_metrics(self, fixed: np.ndarray, warped: np.ndarray) -> dict:
        mi = self._mutual_information(fixed, warped)
        correlation = self._correlation(fixed, warped)
        dice = self._dice_overlap(fixed, warped)
        return {'mi': mi, 'correlation': correlation, 'dice': dice}

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

    def _dice_overlap(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.1) -> float:
        mask1 = img1 > threshold * np.max(img1)
        mask2 = img2 > threshold * np.max(img2)

        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1) + np.sum(mask2)

        return float(2 * intersection / union) if union > 0 else 0.0

    def generate_qc_visualization(self, moving_raster: np.ndarray, atlas_slice: np.ndarray,
                                  warped_image: np.ndarray, warped_row_px: np.ndarray,
                                  warped_col_px: np.ndarray, slice_index: int, output_path: str):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(atlas_slice, cmap=mpl.cm.Reds)
        axes[0].set_title(f'Atlas slice (idx={slice_index})')

        axes[1].imshow(moving_raster, cmap=mpl.cm.Blues)
        axes[1].set_title('Rasterized input')

        axes[2].imshow(atlas_slice, cmap=mpl.cm.Reds, alpha=0.5)
        axes[2].imshow(warped_image, cmap=mpl.cm.Blues, alpha=0.5)
        axes[2].set_title('Overlay (post-registration)')

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


class TransformSearcher:
    """Searches for optimal rotation and/or mirror settings."""

    def __init__(self, config: RegistrationConfig, registrar: ImageRegistrar, qc: QualityController):
        self.config = config
        self.registrar = registrar
        self.qc = qc

    def search(self, x: np.ndarray, y: np.ndarray, atlas_slice: np.ndarray,
               atlas_metadata: ImageMetadata, preprocessor: SpatialOmicsPreprocessor,
               search_rotation: bool, search_mirror: bool) -> tuple[float, bool]:
        """
        Search for optimal rotation and/or mirror settings.
        Returns (best_rotation, best_mirror).
        """
        if search_rotation:
            rotation_angles = list(np.arange(0, 360, self.config.rotation_step))
        else:
            rotation_angles = [self.config.rotation_deg]

        if search_mirror:
            mirror_options = [False, True]
        else:
            mirror_options = [self.config.mirror_x]

        atlas_normalized = self._normalize(atlas_slice)
        atlas_binary = atlas_normalized > 0.1

        n_rotations = len(rotation_angles)
        n_mirrors = len(mirror_options)
        print(f"Searching {n_rotations} rotation(s) × {n_mirrors} mirror setting(s)")

        results = []
        for mirror in mirror_options:
            for angle in rotation_angles:
                ap_trans, dv_trans = preprocessor.transform_coordinates(
                    x, y, rotation_deg=angle, mirror_x=mirror
                )
                ap_centered, dv_centered = preprocessor.center_coordinates(ap_trans, dv_trans)

                atlas_center_x = atlas_metadata.origin[0] + atlas_metadata.shape[1] * atlas_metadata.spacing[0] / 2
                atlas_center_y = atlas_metadata.origin[1] + atlas_metadata.shape[0] * atlas_metadata.spacing[1] / 2

                dv_shifted = dv_centered + atlas_center_x
                ap_shifted = ap_centered + atlas_center_y

                density = preprocessor.rasterize_to_atlas_grid(dv_shifted, ap_shifted, atlas_metadata)

                coverage = np.sum(density > 0)
                if coverage < 10:
                    continue

                raster_normalized = self._normalize(density)
                score = self._compute_similarity(raster_normalized, atlas_normalized, atlas_binary)

                mirror_str = "mirrored" if mirror else "normal"
                print(f"  Angle {angle:5.1f}°, {mirror_str:8s}: score={score:.4f}, coverage={coverage}")

                results.append((angle, mirror, score, coverage, ap_centered, dv_centered))

        if not results:
            print(f"WARNING: No valid transforms found. Using defaults.")
            return self.config.rotation_deg, self.config.mirror_x

        max_coverage = max(r[3] for r in results)
        top_candidates = [r for r in results if r[3] == max_coverage]

        if len(top_candidates) == 1:
            best_angle, best_mirror = top_candidates[0][0], top_candidates[0][1]
            mirror_str = "mirrored" if best_mirror else "normal"
            print(f"Transform search: best = {best_angle}°, {mirror_str} (single max coverage)")
            return best_angle, best_mirror

        print(f"Transform search: {len(top_candidates)} candidates tied with coverage={max_coverage}")
        print("Running affine registration to break tie...")

        fixed_ants = self.registrar.create_ants_image(atlas_slice, atlas_metadata)

        best_angle = top_candidates[0][0]
        best_mirror = top_candidates[0][1]
        best_affine_corr = -1.0

        for angle, mirror, score, coverage, ap_centered, dv_centered in top_candidates:
            X_grid, Y_grid, density = preprocessor.rasterize(dv_centered, ap_centered)

            moving_metadata = ImageMetadata(
                origin=(float(Y_grid[0]), float(X_grid[0])),
                spacing=(self.config.rasterization_resolution, self.config.rasterization_resolution),
                shape=density.shape
            )

            moving_ants = self.registrar.create_ants_image(density, moving_metadata)
            affine_result = self.registrar.affine_register(fixed_ants, moving_ants)
            affine_warped = affine_result['warpedmovout'].numpy()

            metrics = self.qc.compute_metrics(atlas_slice, affine_warped)
            affine_corr = metrics['correlation']

            mirror_str = "mirrored" if mirror else "normal"
            print(f"    Affine {angle:5.1f}°, {mirror_str:8s}: correlation = {affine_corr:.4f}")

            if affine_corr > best_affine_corr:
                best_affine_corr = affine_corr
                best_angle = angle
                best_mirror = mirror

        mirror_str = "mirrored" if best_mirror else "normal"
        print(f"Transform search: best = {best_angle}°, {mirror_str} (affine correlation = {best_affine_corr:.4f})")
        return best_angle, best_mirror

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float64)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return np.zeros_like(img)

    def _compute_similarity(self, moving: np.ndarray, fixed: np.ndarray,
                            fixed_binary: np.ndarray) -> float:
        moving_binary = moving > 0.01
        overlap = np.sum(moving_binary & fixed_binary)
        moving_mass = np.sum(moving_binary)

        if moving_mass < 10:
            return -1.0

        overlap_ratio = overlap / moving_mass

        mask = moving_binary | fixed_binary
        if np.sum(mask) < 10:
            return -1.0

        moving_masked = moving[mask]
        fixed_masked = fixed[mask]

        moving_centered = moving_masked - moving_masked.mean()
        fixed_centered = fixed_masked - fixed_masked.mean()

        denom = np.sqrt(np.sum(moving_centered**2) * np.sum(fixed_centered**2))
        if denom < 1e-10:
            return -1.0

        correlation = np.sum(moving_centered * fixed_centered) / denom
        return overlap_ratio * 0.5 + correlation * 0.5


class RegistrationPipeline:
    def __init__(self, config: RegistrationConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = SpatialOmicsPreprocessor(config)
        self.registrar = ImageRegistrar(config)
        self.qc = QualityController(config)
        self.atlas_loader: Optional[AtlasLoader] = None

    def load_atlas(self, atlas_path: Path):
        self.atlas_loader = AtlasLoader(atlas_path)

    def _determine_quality_flag(self, pre_dice: float, pre_corr: float) -> str:
        if pre_dice > 0.5 and pre_corr > 0.3:
            return "PASS"
        elif pre_dice > 0.3 or pre_corr > 0.2:
            return "REVIEW"
        return "FAIL"

    def run_single_slice(self, omics_df: pd.DataFrame, atlas_slice_index: int,
                         orientation: SliceOrientation, input_name: str) -> RegistrationResult:
        if self.atlas_loader is None:
            raise ValueError("Atlas not loaded. Call load_atlas() first.")

        x, y = self.preprocessor.extract_coordinates(omics_df)
        atlas_slice, atlas_metadata = self.atlas_loader.get_slice(atlas_slice_index, orientation)

        if self.config.search_rotation or self.config.search_mirror:
            searcher = TransformSearcher(self.config, self.registrar, self.qc)
            best_rotation, best_mirror = searcher.search(
                x, y, atlas_slice, atlas_metadata, self.preprocessor,
                search_rotation=self.config.search_rotation,
                search_mirror=self.config.search_mirror
            )
            self.config.rotation_deg = best_rotation
            self.config.mirror_x = best_mirror
            self.preprocessor = SpatialOmicsPreprocessor(self.config)

        ap_trans, dv_trans = self.preprocessor.transform_coordinates(x, y)
        ap_centered, dv_centered = self.preprocessor.center_coordinates(ap_trans, dv_trans)

        X_grid, Y_grid, density = self.preprocessor.rasterize(dv_centered, ap_centered)

        moving_metadata = ImageMetadata(
            origin=(float(Y_grid[0]), float(X_grid[0])),
            spacing=(self.config.rasterization_resolution, self.config.rasterization_resolution),
            shape=density.shape
        )

        fixed_ants = self.registrar.create_ants_image(atlas_slice, atlas_metadata)
        moving_ants = self.registrar.create_ants_image(density, moving_metadata)

        affine_result = self.registrar.affine_register(fixed_ants, moving_ants)
        affine_warped = affine_result['warpedmovout'].numpy()

        pre_syn_metrics = self.qc.compute_metrics(atlas_slice, affine_warped)

        reg_result = self.registrar.register(fixed_ants, moving_ants)

        points_df = pd.DataFrame({'x': ap_centered, 'y': dv_centered})
        warped_points = self.registrar.warp_coordinates(points_df, reg_result['invtransforms'])

        row_px = (warped_points['x'].to_numpy() - fixed_ants.origin[0]) / fixed_ants.spacing[0]
        col_px = (warped_points['y'].to_numpy() - fixed_ants.origin[1]) / fixed_ants.spacing[1]

        warped_image = reg_result['warpedmovout'].numpy()
        post_reg_metrics = self.qc.compute_metrics(atlas_slice, warped_image)

        quality_flag = self._determine_quality_flag(
            pre_syn_metrics['dice'], pre_syn_metrics['correlation']
        )

        quality_metrics = QualityMetrics(
            mutual_information=post_reg_metrics['mi'],
            correlation=post_reg_metrics['correlation'],
            dice_overlap=post_reg_metrics['dice'],
            pre_registration_dice=pre_syn_metrics['dice'],
            pre_registration_correlation=pre_syn_metrics['correlation'],
            registration_converged=post_reg_metrics['mi'] > self.config.similarity_threshold,
            quality_flag=quality_flag
        )

        output_base = f"{input_name}_to_slice_{atlas_slice_index}"

        qc_path = self.output_dir / f"{output_base}_qc.png"
        self.qc.generate_qc_visualization(
            density, atlas_slice, warped_image,
            row_px, col_px, atlas_slice_index,
            str(qc_path)
        )

        output_df = omics_df.copy()
        output_df['ccf_x'] = warped_points['x'].to_numpy() / self.config.coordinate_scale
        output_df['ccf_y'] = warped_points['y'].to_numpy() / self.config.coordinate_scale
        output_df['ccf_z'] = output_df['z'] if 'z' in output_df.columns else 0.0

        output_csv = self.output_dir / f"{output_base}_registered.csv"
        keep_cols = ['cell_label', 'x', 'y', 'z', 'ccf_x', 'ccf_y', 'ccf_z']
        available_cols = [c for c in keep_cols if c in output_df.columns]
        output_df[available_cols].to_csv(output_csv, index=False)

        metrics_path = self.output_dir / f"{output_base}_metrics.txt"
        mirror_str = "Yes" if self.config.mirror_x else "No"
        with open(metrics_path, "w") as f:
            f.write(f"Quality Flag: {quality_metrics.quality_flag}\n")
            f.write(f"Rotation Used: {self.config.rotation_deg:.1f}°\n")
            f.write(f"Mirror X: {mirror_str}\n")
            f.write(f"\nPost-affine (before SyN):\n")
            f.write(f"  Dice: {quality_metrics.pre_registration_dice:.4f}\n")
            f.write(f"  Correlation: {quality_metrics.pre_registration_correlation:.4f}\n")
            f.write(f"\nPost-SyN (final):\n")
            f.write(f"  Mutual Information: {quality_metrics.mutual_information:.4f}\n")
            f.write(f"  Correlation: {quality_metrics.correlation:.4f}\n")
            f.write(f"  Dice: {quality_metrics.dice_overlap:.4f}\n")

        return RegistrationResult(
            warped_image=warped_image,
            forward_transforms=reg_result['fwdtransforms'],
            inverse_transforms=reg_result['invtransforms'],
            warped_coordinates=warped_points,
            quality_metrics=quality_metrics,
            fixed_metadata=atlas_metadata,
            moving_metadata=moving_metadata
        )


def main():
    parser = argparse.ArgumentParser(
        description="Register spatial omics data to developmental brain atlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python registration_pipeline.py --atlas data/aba_nissl.nrrd --omics data/cells.csv --slice 64 --orientation coronal

  python registration_pipeline.py --atlas data/aba_nissl.nrrd --omics data/cells.csv --slice 64 --orientation coronal --search-rotation

  python registration_pipeline.py --atlas data/aba_nissl.nrrd --omics data/cells.csv --slice 64 --orientation coronal --search-mirror

  python registration_pipeline.py --atlas data/aba_nissl.nrrd --omics data/cells.csv --slice 64 --orientation coronal --search-rotation --search-mirror
        """
    )

    parser.add_argument("--atlas", type=Path, required=True, help="Path to atlas NRRD file")
    parser.add_argument("--omics", type=Path, required=True, help="Path to omics CSV file")
    parser.add_argument("--slice", type=int, required=True, help="Atlas slice index")
    parser.add_argument("--orientation", type=str, required=True,
                        choices=["coronal", "sagittal", "axial"], help="Slice orientation")
    parser.add_argument("--output", type=Path, default=Path("registration_results"), help="Output directory")
    parser.add_argument("--rotation", type=float, default=0.0, help="Initial rotation in degrees")
    parser.add_argument("--scale", type=float, nargs=2, default=[0.9, 0.9], help="Scale factors (x, y)")
    parser.add_argument("--mirror", action="store_true", help="Enable X-axis mirroring")
    parser.add_argument("--resolution", type=float, default=50.0, help="Rasterization resolution in µm")
    parser.add_argument("--blur", type=float, default=1.0, help="Gaussian blur sigma")
    parser.add_argument("--search-rotation", action="store_true",
                        help="Search for best initial rotation angle (0-360°)")
    parser.add_argument("--rotation-step", type=float, default=90.0,
                        help="Rotation search step size in degrees (default: 90)")
    parser.add_argument("--search-mirror", action="store_true",
                        help="Search for best mirror setting (normal vs mirrored)")

    args = parser.parse_args()

    config = RegistrationConfig(
        rasterization_resolution=args.resolution,
        gaussian_blur_sigma=args.blur,
        rotation_deg=args.rotation,
        scale=tuple(args.scale),
        mirror_x=args.mirror,
        search_rotation=args.search_rotation,
        rotation_step=args.rotation_step,
        search_mirror=args.search_mirror,
    )

    orientation = SliceOrientation[args.orientation.upper()]

    pipeline = RegistrationPipeline(config=config, output_dir=args.output)
    pipeline.load_atlas(args.atlas)

    omics_df = pipeline.preprocessor.load_data(args.omics)
    input_name = args.omics.stem

    result = pipeline.run_single_slice(
        omics_df=omics_df,
        atlas_slice_index=args.slice,
        orientation=orientation,
        input_name=input_name
    )

    mirror_str = "Yes" if config.mirror_x else "No"
    print(f"\nRegistration complete!")
    print(f"Quality flag: {result.quality_metrics.quality_flag}")
    print(f"Rotation used: {config.rotation_deg:.1f}°")
    print(f"Mirror X: {mirror_str}")
    print(f"\nPre-registration metrics:")
    print(f"  Dice: {result.quality_metrics.pre_registration_dice:.4f}")
    print(f"  Correlation: {result.quality_metrics.pre_registration_correlation:.4f}")
    print(f"\nPost-registration metrics:")
    print(f"  MI: {result.quality_metrics.mutual_information:.4f}")
    print(f"  Correlation: {result.quality_metrics.correlation:.4f}")
    print(f"  Dice: {result.quality_metrics.dice_overlap:.4f}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()