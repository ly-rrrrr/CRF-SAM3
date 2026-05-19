# Sa2VA Data Mirror Layout

This directory preserves the original data path prefix used by the remote laboratory environment:

```text
Sa2VA/data/video_datas/
```

The full datasets are not committed to this repository because raw frames, ground-truth masks, serialized mask dictionaries, and generated prediction masks are too large for ordinary Git hosting.

Instead, this repository keeps:

- the dataset directory skeleton;
- dataset-level README files;
- global sample metadata in `../sample_inventory.json`;
- transfer metadata in `../transfer_mirror_manifest.json`;
- lightweight evaluation and offline-cache metadata under `Sa2VA-main/`.

If you restore full datasets locally, keep the same path structure so the existing configs and scripts can resolve dataset paths with minimal changes.
