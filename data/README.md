# Dataset

The project now uses the MWCD source at `data/final_merged_dataset/`. The
configured labels combine its source folders into eight material groups:
paper, textile, metal, plastic, organic, glass, hazardous, and other.

If you need to restore the dataset, extract the MWCD archive so that its
`final_merged_dataset` directory is directly under `data/`. The project reads
source images without moving or deleting them. Exact duplicate images within a
configured dataset are excluded from evaluation/training by the loader. Data
files are excluded from Git.
