# Data folder

Place your data in this structure, or edit `configs/config.yaml` to point to your own folders.

```text
data/
  FileNames/
    image_classes_OPTOS.csv
    image_classes_LUS.csv
    image_classes_TUS.csv
  images/
    OPTOS_RG/
    LUS/
    TUS/
```

Each CSV should contain at least:

```text
Image Name,Class
example_001.png,Nevus
example_002.png,UM
```

The same subject/order should be represented across the three modality CSVs because the fusion model assumes matching labels across Optos, LUS, and TUS batches.
