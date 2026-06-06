#!/usr/bin/env python
# One-time CPU precompute for data.curation="cluster" (run on a login node, like prepare.py).
# TCGA's tile morphology is long-tailed: common tissue (stroma/fat/background-ish) dominates and drowns out
# rare informative patterns, so a uniform sampler wastes the capped 1M presentations on redundancy. This
# clusters every tile by a coarse colour/spatial descriptor and writes {path, weight=1/cluster_size} so the
# dataloader can draw the 1M presentations balanced across clusters (GenBio-PathFM's diversity-curation idea).
#
#   python curate.py configs/cc-main.yaml          # writes <dataset_dir>/curation_clusters.parquet
#
# Descriptor is an 8x8 RGB thumbnail (192-d) decoded fast via PIL's JPEG draft mode — coarse (gross tissue
# type, not fine morphology) but free and model-free. Covers ALL tiles (both splits); the dataloader looks up
# weights by path for its own split. ~4M tiles -> a few GB RAM and tens of minutes single-threaded.
import io
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pathlib import Path
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

K = 256
cfg = yaml.safe_load(open(os.path.expandvars(sys.argv[1])))
dataset_dir = Path(os.path.expandvars(cfg["data"]["dataset_dir"]))
shards = sorted(dataset_dir.glob("shard-*.parquet"))
assert shards, f"no shard-*.parquet under {dataset_dir}"

paths, feats = [], []
for shard in shards:
    table = pq.read_table(str(shard), columns=["path", "jpeg"], memory_map=True)
    for p, jpeg_bytes in zip(table["path"].to_pylist(), table["jpeg"].to_pylist()):
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            img.draft("RGB", (32, 32))  # decode the JPEG at ~1/8 scale — much faster than full decode
            thumb = np.asarray(img.convert("RGB").resize((8, 8))) / 255.0
        paths.append(p)
        feats.append(thumb.reshape(-1))
    print(f"  {shard.name}: {len(paths)} tiles so far", flush=True)

X = np.asarray(feats, dtype=np.float32)
labels = MiniBatchKMeans(n_clusters=K, batch_size=8192, n_init=3, random_state=0).fit_predict(X)
weight = 1.0 / np.bincount(labels, minlength=K)[labels]
weight = weight / weight.mean()
out = dataset_dir / "curation_clusters.parquet"
pq.write_table(pa.table({"path": paths, "weight": weight.astype(np.float64)}), out)
print(f"wrote {len(paths)} tile weights across {K} clusters -> {out}")
