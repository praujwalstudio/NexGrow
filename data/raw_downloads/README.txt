NexGrow Dataset Staging Area
============================
Extract downloaded ZIPs into numbered subfolders here:

  dataset1_mendeley/   <- Shamim Ripon (Bacterial_Blight, Alternaria, Fusarium, Verticillium, Healthy)
  dataset2_clcud/      <- janmejaybhoi CLCuD (CLCuD, Leaf_Curl)
  dataset3_crop/       <- Cotton Crop Disease Detection (Target_Spot, Anthracnose, Grey_Mildew)
  dataset4_cotad/      <- COT-AD (Red_Rust, Cercospora_Leaf_Spot, Root_Rot)

Then run:
  python scripts/dataset_mapper.py --source data/raw_downloads --output data/raw --dry-run
  python scripts/dataset_mapper.py --source data/raw_downloads --output data/raw
