"""
prepare_data.py

Organises raw dataset folders into the two-stage class structure:

Stage 1 (coarse):
    data/raw/stage1/gunshot/     <- Zenodo + UrbanSound8K classID 6
    data/raw/stage1/siren/       <- sireNNet (police+ambulance+firetruck) + UrbanSound8K classID 8
    data/raw/stage1/background/  <- sireNNet traffic + UrbanSound8K all other classes

Stage 2 (fine, siren types only):
    data/raw/stage2/police/      <- sireNNet/police
    data/raw/stage2/ambulance/   <- sireNNet/ambulance
    data/raw/stage2/firetruck/   <- sireNNet/firetruck
    data/raw/stage2/unknown/     <- UrbanSound8K classID 8 (type not labelled)

Run:
    docker compose run app python src/prepare_data.py
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SOURCE PATHS
# ---------------------------------------------------------------------------

URBAN_DIR  = "data/raw/datasets/archive"
ZENODO_DIR = "data/raw/datasets/edge-collected-gunshot-audio/edge-collected-gunshot-audio"
SIREN_DIR  = "data/raw/datasets/sireNNet"

# ---------------------------------------------------------------------------
# TARGET DIRECTORIES
# ---------------------------------------------------------------------------

STAGE1_CLASSES = ["gunshot", "siren", "background"]
STAGE2_CLASSES = ["police", "ambulance", "firetruck", "unknown"]

for c in STAGE1_CLASSES:
    os.makedirs(os.path.join("data/raw/stage1", c), exist_ok=True)
for c in STAGE2_CLASSES:
    os.makedirs(os.path.join("data/raw/stage2", c), exist_ok=True)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def collect_wavs(folder):
    result = []
    for dirpath, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".wav"):
                result.append(os.path.join(dirpath, f))
    return result


def copy_file(src, dst_folder, new_name):
    dst = os.path.join(dst_folder, new_name)
    if os.path.exists(dst):
        base, ext = os.path.splitext(new_name)
        i = 1
        while os.path.exists(dst):
            dst = os.path.join(dst_folder, f"{base}_{i}{ext}")
            i += 1
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# 1. URBANSOUND8K
# ---------------------------------------------------------------------------

def process_urbansound8k():
    print("\n--- UrbanSound8K ---")
    csv_path = os.path.join(URBAN_DIR, "UrbanSound8K.csv")
    df = pd.read_csv(csv_path)

    all_wavs = collect_wavs(URBAN_DIR)
    fname_to_path = {os.path.basename(p): p for p in all_wavs}

    s1_counts = {"gunshot": 0, "siren": 0, "background": 0}
    s2_unknown = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="UrbanSound8K"):
        cid   = int(row["classID"])
        fname = row["slice_file_name"]
        fsid  = int(row["fsID"])
        src   = fname_to_path.get(fname)
        if src is None:
            continue

        new_name = f"us8k_{fsid}_{fname}"

        if cid == 6:
            # gunshot -> stage1/gunshot only
            copy_file(src, "data/raw/stage1/gunshot", new_name)
            s1_counts["gunshot"] += 1

        elif cid == 8:
            # siren (type unknown) -> stage1/siren + stage2/unknown
            copy_file(src, "data/raw/stage1/siren", new_name)
            copy_file(src, "data/raw/stage2/unknown", new_name)
            s1_counts["siren"] += 1
            s2_unknown += 1

        else:
            # everything else -> stage1/background
            copy_file(src, "data/raw/stage1/background", new_name)
            s1_counts["background"] += 1

    for k, v in s1_counts.items():
        print(f"  stage1/{k}: {v}")
    print(f"  stage2/unknown: {s2_unknown}")


# ---------------------------------------------------------------------------
# 2. ZENODO GUNSHOT
# ---------------------------------------------------------------------------

def process_zenodo():
    print("\n--- Zenodo Gunshot ---")
    all_wavs   = collect_wavs(ZENODO_DIR)
    mono_files = [
        p for p in all_wavs
        if "_chan" not in os.path.basename(p).lower()
    ]
    print(f"  mono files: {len(mono_files)}")

    for src in tqdm(mono_files, desc="Zenodo"):
        new_name = "zenodo_" + os.path.basename(src)
        copy_file(src, "data/raw/stage1/gunshot", new_name)

    print(f"  stage1/gunshot: {len(mono_files)}")


# ---------------------------------------------------------------------------
# 3. SIRENNET
# ---------------------------------------------------------------------------

SIRENNET_FOLDERS = ["police", "ambulance", "firetruck"]

def process_sirennet():
    print("\n--- sireNNet ---")

    for folder_name in SIRENNET_FOLDERS:
        src_folder = os.path.join(SIREN_DIR, folder_name)
        if not os.path.isdir(src_folder):
            print(f"  [WARNING] not found: {src_folder}")
            continue

        wavs = collect_wavs(src_folder)
        for src in tqdm(wavs, desc=f"sireNNet/{folder_name}"):
            new_name = f"sirennet_{folder_name}_" + os.path.basename(src)
            # goes to stage1/siren AND stage2/<type>
            copy_file(src, "data/raw/stage1/siren", new_name)
            copy_file(src, f"data/raw/stage2/{folder_name}", new_name)

        print(f"  stage1/siren += {len(wavs)}  |  stage2/{folder_name}: {len(wavs)}")

    # traffic -> stage1/background only
    traffic_dir = os.path.join(SIREN_DIR, "traffic")
    if os.path.isdir(traffic_dir):
        wavs = collect_wavs(traffic_dir)
        for src in tqdm(wavs, desc="sireNNet/traffic"):
            new_name = "sirennet_traffic_" + os.path.basename(src)
            copy_file(src, "data/raw/stage1/background", new_name)
        print(f"  stage1/background += {len(wavs)}")


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_summary():
    print("\n--- Summary ---")
    print("Stage 1:")
    for c in STAGE1_CLASSES:
        folder = os.path.join("data/raw/stage1", c)
        n = len([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
        print(f"  {c:<12}: {n:>5} files")
    print("Stage 2:")
    for c in STAGE2_CLASSES:
        folder = os.path.join("data/raw/stage2", c)
        n = len([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
        print(f"  {c:<12}: {n:>5} files")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    missing = []
    for label, path in [
        ("UrbanSound8K", URBAN_DIR),
        ("Zenodo",       ZENODO_DIR),
        ("sireNNet",     SIREN_DIR),
    ]:
        if not os.path.exists(path):
            missing.append(f"  {label}: {path}")

    if missing:
        print("ERROR - folders not found:")
        for m in missing:
            print(m)
        exit(1)

    process_urbansound8k()
    process_zenodo()
    process_sirennet()
    print_summary()