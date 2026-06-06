#!/usr/bin/env python
# One-time precompute for data.curation="cluster". MUST run on a COMPUTE node, not a login node: it decodes
# every TCGA tile (~4M) and a login node's CPU-time ulimit kills it partway. It only reads local /scratch
# shards (no internet), so a plain CPU allocation works:
#   sbatch --account=<acct> --cpus-per-task=16 --mem=64G --time=1:00:00 --output=/scratch/%u/curate-%j.out \
#     --wrap "cd $PWD && module load python/3.12 && source .venv/bin/activate && python curate.py configs/cc-main.yaml"
#
# TCGA tile morphology is long-tailed: common tissue dominates and drowns out rare informative patterns, so a
# uniform sampler wastes the capped 1M presentations on redundancy. This clusters every tile by a coarse
# colour/spatial descriptor and writes {path, weight=1/cluster_size} so the dataloader can draw the 1M
# presentations balanced across clusters (GenBio-PathFM's diversity-curation idea). Descriptor = an 8x8 RGB
# thumbnail (192-d) decoded fast via PIL JPEG draft mode — coarse (gross tissue type) but free and model-free.
import io
import os
import sys

# Cap BLAS/OMP threads BEFORE importing numpy/sklearn (it only takes effect pre-import): Alliance's OpenBLAS
# is built for <=128 threads and SEGFAULTS when the final k-means runs on a whole 192-core Trillium node.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
os.environ.setdefault("OMP_NUM_THREADS", "64")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

K = 256


# Decode one shard's tiles to (paths, 192-d descriptors); runs in a worker process.
def descriptors_for_shard(shard_path):
    table = pq.read_table(shard_path, columns=["path", "jpeg"], memory_map=True)
    paths, feats = [], []
    for p, jpeg_bytes in zip(table["path"].to_pylist(), table["jpeg"].to_pylist()):
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            img.draft("RGB", (32, 32))  # decode the JPEG at ~1/8 scale — much faster than full decode
            thumb = np.asarray(img.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
        paths.append(p)
        feats.append(thumb.reshape(-1))
    return paths, np.asarray(feats, dtype=np.float32)


if __name__ == "__main__":
    cfg = yaml.safe_load(open(os.path.expandvars(sys.argv[1])))
    dataset_dir = Path(os.path.expandvars(cfg["data"]["dataset_dir"]))
    shards = [str(s) for s in sorted(dataset_dir.glob("shard-*.parquet"))]
    assert shards, f"no shard-*.parquet under {dataset_dir}"
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))

    paths, feats = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (sp, sf) in enumerate(pool.map(descriptors_for_shard, shards)):
            paths.extend(sp)
            feats.append(sf)
            print(f"  shard {i + 1}/{len(shards)}: {len(paths)} tiles", flush=True)
    X = np.concatenate(feats)

    labels = MiniBatchKMeans(n_clusters=K, batch_size=8192, n_init=3, random_state=0).fit_predict(X)
    weight = 1.0 / np.bincount(labels, minlength=K)[labels]
    weight = weight / weight.mean()
    out = dataset_dir / "curation_clusters.parquet"
    pq.write_table(pa.table({"path": paths, "weight": weight.astype(np.float64)}), out)
    print(f"wrote {len(paths)} tile weights across {K} clusters -> {out}")
