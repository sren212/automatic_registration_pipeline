#!/bin/bash

for x in $(seq 30 30 330); do
    echo "Processing rotation $x..."
    python registration_pipeline.py \
        --atlas data/aba_nissl.nrrd \
        --omics "data/abca3_slice_13_cell_metadata_rot${x}.csv" \
        --slice 64 \
        --output "no_search_registration_results/rot${x}"
done

echo "All rotations complete!"