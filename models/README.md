# Model artifacts

Training writes a uniquely named `.keras` model and JSON metadata file here.
Large model artifacts are excluded from Git. Keep the metadata beside its
model: it records the ordered class labels and preprocessing settings needed
for correct inference.

