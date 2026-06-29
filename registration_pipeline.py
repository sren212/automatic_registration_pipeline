#!/usr/bin/env python3
"""
Automated image registration pipeline for aligning developmental mouse brain
spatial omics data to KimLab 3D developmental Brain CCF atlases using ANTsPy.
"""

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import ants
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


class SliceOrientation(Enum):
    CORONAL = "coronal"
    SAGITTAL = "sagittal"
    AXIAL = "axial"


class RegistrationMetric(Enum):
    MUTUAL_INFORMATION = "mattes"
    CORRELATION = "gc"


@dataclass
class ImageMetadata:
    source_path: Path
    slice_index: int
    orientation: SliceOrientation
    original_shape: Tuple[int, int]
    resolution_um: float


@dataclass
class RegistrationConfig:
    rotation_deg: float = 0.0
    scale: Tuple[float, float] = (0.9, 0.9)
    mirror_x: bool = False
    resolution_um: float = 50.0
    blur_sigma: float = 1.0
    search_rotation: bool = False
    rotation_range: Tuple[float, float] = (0, 360)
    rotation_step: float = 15.0
    search_mirror: bool = False


@dataclass
class QualityMetrics:
    dice_coefficient: float
    correlation: float
    pre_registration_dice: float
    pre_registration_correlation: float
    flag: str


@dataclass
class RegistrationResult:
    metadata: ImageMetadata
    config: RegistrationConfig
    quality: QualityMetrics
    transform: Any
    warped_image: np.ndarray
    detected_rotation: Optional[float] = None
    detected_mirror: Optional[bool] = None


class AtlasLoader:
    """Handles loading and slicing of 3D atlas volumes."""

    def __init__(self, atlas_path: Path):
        self.atlas_path = atlas_path
        self.volume = self._load_volume()

    def _load_volume(self) -> np.ndarray:
        if self.atlas_path.suffix == '.npy':
            return np.load(self.atlas_path)
        elif self.atlas_path.suffix in ['.nii', '.gz']:
            return ants.image_read(str(self.atlas_path)).numpy()
        else:
            raise ValueError(f"Unsupported atlas format: {self.atlas_path.suffix}")

    def get_slice(self, index: int, orientation: SliceOrientation) -> np.ndarray:
        if orientation == SliceOrientation.CORONAL:
            return self.volume[index, :, :]
        elif orientation == SliceOrientation.SAGITTAL:
            return self.volume[:, index, :]
        elif orientation == SliceOrientation.AXIAL:
            return self.volume[:, :, index]


class SpatialOmicsPreprocessor:
    """Preprocesses spatial omics data for registration."""

    def __init__(self, config: RegistrationConfig):
        self.config = config


def extract_coordinates(omics_path: Path) -> np.ndarray:
    """Extract spatial coordinates from omics file. Returns (y, x) to match atlas convention."""
    df = pd.read_csv(omics_path)
    x_col = next(c for c in df.columns if c.lower() in ['x', 'x_um', 'x_coord'])
    y_col = next(c for c in df.columns if c.lower() in ['y', 'y_um', 'y_coord'])
    return np.column_stack([df[y_col].values, df[x_col].values])


def transform_coordinates(coords: np.ndarray, config: RegistrationConfig) -> np.ndarray:
    """Apply rotation, scale, and mirror transformations to coordinates."""
    transformed = coords.copy().astype(float)

    # Center coordinates
    centroid = transformed.mean(axis=0)
    transformed -= centroid

    # Apply rotation
    theta = np.radians(config.rotation_deg)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    transformed = transformed @ rotation_matrix.T

    # Apply scale
    transformed[:, 0] *= config.scale[0]
    transformed[:, 1] *= config.scale[1]

    # Apply mirror
    if config.mirror_x:
        transformed[:, 1] = -transformed[:, 1]

    # Shift to positive coordinates
    transformed -= transformed.min(axis=0)

    return transformed


def rasterize(
    coords: np.ndarray,
    atlas_shape: Tuple[int, int],
    blur_sigma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert point coordinates to density image matching atlas dimensions."""
    ap_coords, dv_coords = coords[:, 0], coords[:, 1]

    ap_min, ap_max = ap_coords.min(), ap_coords.max()
    dv_min, dv_max = dv_coords.min(), dv_coords.max()

    ap_normalized = (ap_coords - ap_min) / (ap_max - ap_min) * (atlas_shape[0] - 1)
    dv_normalized = (dv_coords - dv_min) / (dv_max - dv_min) * (atlas_shape[1] - 1)

    density, _, _ = np.histogram2d(
        ap_normalized, dv_normalized,
        bins=[atlas_shape[0], atlas_shape[1]],
        range=[[0, atlas_shape[0]], [0, atlas_shape[1]]]
    )

    if blur_sigma > 0:
        density = gaussian_filter(density, sigma=blur_sigma)

    Y_grid, X_grid = np.mgrid[0:atlas_shape[0], 0:atlas_shape[1]]

    return X_grid, Y_grid, density


class ImageRegistrar:
    """Performs image registration using ANTsPy."""

    def __init__(self, config: RegistrationConfig):
        self.config = config

    def affine_register(
        self,
        moving: ants.ANTsImage,
        fixed: ants.ANTsImage
    ) -> Dict[str, Any]:
        """Perform affine-only registration."""
        result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform='Affine',
            metric='mattes'
        )
        return {
            'warped': result['warpedmovout'],
            'transform': result['fwdtransforms']
        }

    def register(
        self,
        moving: ants.ANTsImage,
        fixed: ants.ANTsImage
    ) -> Dict[str, Any]:
        """Perform full SyNRA registration."""
        result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform='SyNRA',
            metric='mattes'
        )
        return {
            'warped': result['warpedmovout'],
            'transform': result['fwdtransforms']
        }


class QualityController:
    """Evaluates registration quality and assigns pass/review/fail flags."""

    def __init__(
        self,
        dice_pass: float = 0.5,
        dice_review: float = 0.3,
        corr_pass: float = 0.3,
        corr_review: float = 0.2
    ):
        self.dice_pass = dice_pass
        self.dice_review = dice_review
        self.corr_pass = corr_pass
        self.corr_review = corr_review

    def compute_metrics(
        self,
        warped: ants.ANTsImage,
        fixed: ants.ANTsImage,
        pre_dice: float = 0.0,
        pre_corr: float = 0.0
    ) -> QualityMetrics:
        warped_np = warped.numpy()
        fixed_np = fixed.numpy()

        warped_binary = warped_np > np.percentile(warped_np, 10)
        fixed_binary = fixed_np > 0

        intersection = np.sum(warped_binary & fixed_binary)
        union = np.sum(warped_binary) + np.sum(fixed_binary)
        dice = 2 * intersection / union if union > 0 else 0.0

        warped_flat = warped_np.flatten()
        fixed_flat = fixed_np.flatten()
        if warped_flat.std() > 0 and fixed_flat.std() > 0:
            correlation = np.corrcoef(warped_flat, fixed_flat)[0, 1]
        else:
            correlation = 0.0

        flag = self._assign_flag(pre_dice, pre_corr)

        return QualityMetrics(
            dice_coefficient=dice,
            correlation=correlation,
            pre_registration_dice=pre_dice,
            pre_registration_correlation=pre_corr,
            flag=flag
        )

    def _assign_flag(self, dice: float, corr: float) -> str:
        if dice > self.dice_pass and corr > self.corr_pass:
            return "PASS"
        elif dice > self.dice_review or corr > self.corr_review:
            return "REVIEW"
        else:
            return "FAIL"


class TransformSearcher:
    """Search for optimal rotation and mirror transformations."""

    def __init__(
        self,
        config: RegistrationConfig,
        registrar: ImageRegistrar,
        qc: QualityController
    ):
        self.config = config
        self.registrar = registrar
        self.qc = qc

    def search(
        self,
        coords: np.ndarray,
        atlas_slice: np.ndarray,
        rotation_range: Tuple[float, float] = (0, 360),
        rotation_step: float = 15.0,
        search_mirror: bool = False
    ) -> Tuple[float, bool]:
        """
        Search for best rotation and mirror combination.

        Returns:
            Tuple of (best_rotation_deg, best_mirror_x)
        """
        rotations = np.arange(rotation_range[0], rotation_range[1], rotation_step)
        if len(rotations) == 0:
            rotations = np.array([self.config.rotation_deg])

        mirror_options = [False, True] if search_mirror else [self.config.mirror_x]

        candidates = []
        for rot in rotations:
            for mirror in mirror_options:
                coverage = self._compute_coverage(coords, atlas_slice, rot, mirror)
                candidates.append((float(rot), mirror, coverage))

        max_coverage = max(c[2] for c in candidates)

        tolerance = 0.01
        best_candidates = [(r, m) for r, m, c in candidates if c >= max_coverage - tolerance]

        if len(best_candidates) == 1:
            return best_candidates[0]

        return self._tiebreak_with_affine(coords, atlas_slice, best_candidates)

    def _compute_coverage(
        self,
        coords: np.ndarray,
        atlas_slice: np.ndarray,
        rotation_deg: float,
        mirror_x: bool
    ) -> float:
        """Compute fraction of transformed points falling within atlas mask."""
        temp_config = RegistrationConfig(
            rotation_deg=rotation_deg,
            scale=self.config.scale,
            mirror_x=mirror_x,
            resolution_um=self.config.resolution_um,
            blur_sigma=self.config.blur_sigma
        )

        transformed = transform_coordinates(coords, temp_config)

        atlas_shape = atlas_slice.shape
        atlas_mask = atlas_slice > 0

        _, _, density = rasterize(transformed, atlas_shape, temp_config.blur_sigma)

        density_mask = density > 0
        overlap = np.sum(density_mask & atlas_mask)
        total_density = np.sum(density_mask)

        return overlap / total_density if total_density > 0 else 0.0

    def _tiebreak_with_affine(
        self,
        coords: np.ndarray,
        atlas_slice: np.ndarray,
        candidates: List[Tuple[float, bool]]
    ) -> Tuple[float, bool]:
        """Use affine registration correlation to break ties."""
        best_corr = -1.0
        best_candidate = candidates[0]

        atlas_ants = ants.from_numpy(atlas_slice.astype(np.float32))

        for rotation_deg, mirror_x in candidates:
            temp_config = RegistrationConfig(
                rotation_deg=rotation_deg,
                scale=self.config.scale,
                mirror_x=mirror_x,
                resolution_um=self.config.resolution_um,
                blur_sigma=self.config.blur_sigma
            )

            transformed = transform_coordinates(coords, temp_config)
            _, _, density = rasterize(transformed, atlas_slice.shape, temp_config.blur_sigma)

            omics_ants = ants.from_numpy(density.astype(np.float32))

            try:
                result = self.registrar.affine_register(omics_ants, atlas_ants)
                metrics = self.qc.compute_metrics(result['warped'], atlas_ants)

                if metrics.correlation > best_corr:
                    best_corr = metrics.correlation
                    best_candidate = (rotation_deg, mirror_x)
            except Exception:
                continue

        return best_candidate


class RegistrationPipeline:
    """Main pipeline orchestrating the registration workflow."""

    def __init__(
        self,
        atlas_path: Path,
        config: RegistrationConfig,
        orientation: SliceOrientation = SliceOrientation.CORONAL
    ):
        self.atlas_loader = AtlasLoader(atlas_path)
        self.preprocessor = SpatialOmicsPreprocessor(config)
        self.registrar = ImageRegistrar(config)
        self.qc = QualityController()
        self.config = config
        self.orientation = orientation

    def run_single_slice(self, omics_path: Path, slice_idx: int) -> RegistrationResult:
        coords = extract_coordinates(omics_path)
        atlas_slice = self.atlas_loader.get_slice(slice_idx, self.orientation)

        rotation_deg = self.config.rotation_deg
        mirror_x = self.config.mirror_x
        detected_rotation = None
        detected_mirror = None

        if self.config.search_rotation or self.config.search_mirror:
            searcher = TransformSearcher(self.config, self.registrar, self.qc)

            if self.config.search_rotation:
                rotation_range = self.config.rotation_range
                rotation_step = self.config.rotation_step
            else:
                rotation_range = (self.config.rotation_deg, self.config.rotation_deg + 1)
                rotation_step = 360.0

            rotation_deg, mirror_x = searcher.search(
                coords, atlas_slice,
                rotation_range=rotation_range,
                rotation_step=rotation_step,
                search_mirror=self.config.search_mirror
            )
            detected_rotation = rotation_deg
            detected_mirror = mirror_x

        working_config = RegistrationConfig(
            rotation_deg=rotation_deg,
            scale=self.config.scale,
            mirror_x=mirror_x,
            resolution_um=self.config.resolution_um,
            blur_sigma=self.config.blur_sigma
        )

        transformed = transform_coordinates(coords, working_config)
        X_grid, Y_grid, density = rasterize(
            transformed, atlas_slice.shape, working_config.blur_sigma
        )

        omics_ants = ants.from_numpy(density.astype(np.float32))
        atlas_ants = ants.from_numpy(atlas_slice.astype(np.float32))

        affine_result = self.registrar.affine_register(omics_ants, atlas_ants)
        pre_metrics = self.qc.compute_metrics(affine_result['warped'], atlas_ants)

        result = self.registrar.register(omics_ants, atlas_ants)

        quality = self.qc.compute_metrics(
            result['warped'], atlas_ants,
            pre_dice=pre_metrics.dice_coefficient,
            pre_corr=pre_metrics.correlation
        )

        metadata = ImageMetadata(
            source_path=omics_path,
            slice_index=slice_idx,
            orientation=self.orientation,
            original_shape=density.shape,
            resolution_um=working_config.resolution_um
        )

        return RegistrationResult(
            metadata=metadata,
            config=working_config,
            quality=quality,
            transform=result['transform'],
            warped_image=result['warped'].numpy(),
            detected_rotation=detected_rotation,
            detected_mirror=detected_mirror
        )

    def run_batch(
        self,
        omics_paths: List[Path],
        slice_indices: List[int]
    ) -> List[RegistrationResult]:
        results = []
        for omics_path, slice_idx in zip(omics_paths, slice_indices):
            result = self.run_single_slice(omics_path, slice_idx)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Register spatial omics data to developmental brain atlas'
    )
    parser.add_argument('--atlas', type=Path, required=True,
                        help='Path to 3D atlas volume (.npy or .nii)')
    parser.add_argument('--omics', type=Path, required=True,
                        help='Path to spatial omics CSV file')
    parser.add_argument('--slice', type=int, required=True,
                        help='Atlas slice index')
    parser.add_argument('--orientation', type=str, default='coronal',
                        choices=['coronal', 'sagittal', 'axial'],
                        help='Slice orientation (default: coronal)')
    parser.add_argument('--output', type=Path, default=Path('output'),
                        help='Output directory (default: output)')
    parser.add_argument('--rotation', type=float, default=0.0,
                        help='Initial rotation in degrees (default: 0.0)')
    parser.add_argument('--scale', type=float, nargs=2, default=[0.9, 0.9],
                        help='Scale factors for y and x (default: 0.9 0.9)')
    parser.add_argument('--mirror', action='store_true',
                        help='Apply mirror transformation along x-axis')
    parser.add_argument('--resolution', type=float, default=50.0,
                        help='Resolution in micrometers (default: 50.0)')
    parser.add_argument('--blur', type=float, default=1.0,
                        help='Gaussian blur sigma (default: 1.0)')
    parser.add_argument('--search-rotation', action='store_true',
                        help='Search for optimal rotation angle')
    parser.add_argument('--rotation-step', type=float, default=15.0,
                        help='Rotation search step in degrees (default: 15.0)')
    parser.add_argument('--search-mirror', action='store_true',
                        help='Search both mirrored and non-mirrored orientations')

    args = parser.parse_args()

    config = RegistrationConfig(
        rotation_deg=args.rotation,
        scale=tuple(args.scale),
        mirror_x=args.mirror,
        resolution_um=args.resolution,
        blur_sigma=args.blur,
        search_rotation=args.search_rotation,
        rotation_step=args.rotation_step,
        search_mirror=args.search_mirror
    )

    orientation = SliceOrientation(args.orientation)

    pipeline = RegistrationPipeline(args.atlas, config, orientation)
    result = pipeline.run_single_slice(args.omics, args.slice)

    args.output.mkdir(parents=True, exist_ok=True)

    np.save(args.output / 'warped.npy', result.warped_image)

    summary = {
        'source': str(result.metadata.source_path),
        'slice_index': result.metadata.slice_index,
        'orientation': result.metadata.orientation.value,
        'rotation_deg': result.config.rotation_deg,
        'mirror_x': result.config.mirror_x,
        'scale': result.config.scale,
        'dice': result.quality.dice_coefficient,
        'correlation': result.quality.correlation,
        'pre_dice': result.quality.pre_registration_dice,
        'pre_correlation': result.quality.pre_registration_correlation,
        'flag': result.quality.flag,
        'detected_rotation': result.detected_rotation,
        'detected_mirror': result.detected_mirror
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(args.output / 'summary.csv', index=False)

    print(f"Registration complete: {result.quality.flag}")
    print(f"  Dice: {result.quality.dice_coefficient:.3f}")
    print(f"  Correlation: {result.quality.correlation:.3f}")
    print(f"  Pre-registration Dice: {result.quality.pre_registration_dice:.3f}")
    print(f"  Pre-registration Correlation: {result.quality.pre_registration_correlation:.3f}")
    print(f"  Rotation: {result.config.rotation_deg:.1f}°")
    print(f"  Mirror: {result.config.mirror_x}")
    if result.detected_rotation is not None:
        print(f"  Detected rotation: {result.detected_rotation:.1f}°")
    if result.detected_mirror is not None:
        print(f"  Detected mirror: {result.detected_mirror}")


if __name__ == '__main__':
    main()