from pathlib import Path

import numpy as np
import pandas as pd


def rotate_merfish_csv(input_csv, angle_degrees_clockwise):
    input_csv = Path(input_csv)
    angle_str = f"{angle_degrees_clockwise:g}"
    output_csv = input_csv.with_name(
        f"{input_csv.stem}_rot{angle_str}{input_csv.suffix}"
    )

    df = pd.read_csv(input_csv)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    # Center of rotation
    cx = (x.min() + x.max()) / 2
    cy = (y.min() + y.max()) / 2

    # Shift to origin
    x_shift = x - cx
    y_shift = y - cy

    # Clockwise rotation
    theta = np.deg2rad(angle_degrees_clockwise)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_rot = x_shift * cos_t + y_shift * sin_t
    y_rot = -x_shift * sin_t + y_shift * cos_t

    # Shift back to original center
    df["x"] = x_rot + cx
    df["y"] = y_rot + cy

    df.to_csv(output_csv, index=False)
    print(f"Saved rotated coordinates to:\n{output_csv}")

    return output_csv

for i in range(30, 360, 30):
    rotate_merfish_csv(
        "data/abca3_slice_13_cell_metadata.csv",
        angle_degrees_clockwise=i,
    )