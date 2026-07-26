# Mapillary Collector

Collects a geolocated street-level image dataset from Mapillary, packages it into
uniform WebDataset tar shards, and uploads them to a Hugging Face dataset repo.

Built to run unattended for days on a laptop. Interrupt it any time; rerun resumes.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

No GDAL, no geopandas. Country boundaries are fetched as GeoJSON and parsed with
shapely, so installation is pure wheels on every platform.

## Configure

```bash
cp .env.example .env
```

Fill in `MAPILLARY_TOKEN` and `HF_TOKEN` (the HF token needs **write** scope).

## Run

```bash
mapillary doctor      # pre-flight check
mapillary quota       # preview how many images each country will get
mapillary run         # collect (resumable)
```

Leave it running:

```bash
tmux new -s mly
mapillary run
# ctrl-b then d to detach; tmux attach -t mly to come back
```

## How it works

**Coverage-first.** For each country it reads Mapillary's coverage tiles (coarse
zoom to find where images exist, fine zoom for the actual image ids), then spends
graph-API calls only on images already known to exist.

**Proportional quota.** Each country's target scales with its coverage, sublinearly:

```
quota = clamp(200, 35 * leaf_tiles^0.5, 5000)
```

Sublinear on purpose: a linear quota would make a few coverage-rich countries most
of the dataset, and the model would learn to guess the majority country instead of
reading the image.

**Uniform sampling.** Candidates are ordered round-robin across tiles: image #1 from
every covered tile in the country before image #2 from any of them. Combined with an
11m coordinate dedupe and a 5-per-sequence cap, samples stay spread out.

**Uniform shards.** Validated images land in a staging directory as loose files
(written temp-then-rename, so nothing on disk is ever partial). Only when staging
holds a full `shard_size` do they get packed into a tar. A crash leaves complete
loose files, and the next run keeps filling toward a full shard — so every shard is
exactly 1000 samples.

## Commands

```bash
mapillary run --country Japan --country Brazil   # only these
mapillary run --exclude Antarctica               # skip these
mapillary run --workers 12                       # more parallelism
mapillary run --dry-run                          # collect, never upload
mapillary status                                 # progress summary
mapillary countries                              # per-country results and timings
mapillary verify --remote                        # check shards locally and on the hub
mapillary finalize                               # ship leftover staged images
mapillary reset --confirm RESET                  # back up and wipe local state
```

## Where things live

Everything is under `~/.mapillary_collector` (override with `--data-dir`):

```
state.sqlite   progress, dedupe index, tile cache, candidate lists
staging/       validated images waiting for a full shard
shards/        packed tars awaiting upload (deleted once verified on the hub)
cache/         Natural Earth boundaries
collector.log  full run log
```

Hugging Face is the durable store: once a shard is verified there, local copies are
disposable.

## Training on the result

```python
import webdataset as wds

url = ("https://huggingface.co/datasets/skuangster/Mapillary_Dataset/"
       "resolve/main/images/shard-{000000..000042}.tar")
dataset = (wds.WebDataset(url)
           .decode("rgb")
           .to_tuple("jpg", "json")
           .batched(32))
```

Streams shard by shard — no full download, works on CPU, GPU or TPU.

Each sample's json carries: `lat`, `lng`, `coord_source`, `country`, `iso3`,
`continent`, `compass`, `computed_compass`, `captured_at`, `quality`, `sequence`,
`camera_type`, `width`, `height`.

## Failure handling

| Failure | Behaviour |
|---|---|
| Ctrl-C / SIGTERM | finishes current image, keeps staging, drains uploads, exits clean |
| Network drop | retries with jittered backoff, honours `Retry-After` |
| 429 rate limit | slows the whole pipeline adaptively, recovers when the API does |
| Corrupt / truncated image | rejected by validation, never enters a shard |
| Crash mid-write | temp-then-rename means no partial files; orphans cleaned at startup |
| Crash mid-upload | startup checks the hub, never uploads the same shard twice |
| Disk full | drains uploads to reclaim space, then stops with a clear message |
| Laptop sleep | connections reopen on wake; failed requests retry |
