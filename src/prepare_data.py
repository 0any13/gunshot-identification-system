"""
prepare_data.py

Reads from already-extracted dataset folders and copies files into
data/raw/ with the correct class structure.

Run from your project root:
    docker compose run app python src/prepare_data.py
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PATHS - exactly as they appear in your Downloads folder
# ---------------------------------------------------------------------------

URBAN_DIR  = "data/raw/datasets/archive"
ZENODO_DIR = "data/raw/datasets/edge-collected-gunshot-audio/edge-collected-gunshot-audio"
SIREN_DIR  = "data/raw/datasets/sireNNet"

PROJECT_ROOT = "."
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")

# ---------------------------------------------------------------------------
# TARGET CLASS FOLDERS
# ---------------------------------------------------------------------------

CLASSES = [
    "gunshot",
    "police_siren",
    "ambulance_siren",
    "firetruck_siren",
    "siren_unknown",
    "background",
]

for c in CLASSES:
    os.makedirs(os.path.join(RAW_DIR, c), exist_ok=True)

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
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find UrbanSound8K.csv at {csv_path}")

    df = pd.read_csv(csv_path)

    class_map = {}
    for _, row in df.iterrows():
        cid   = int(row["classID"])
        fname = row["slice_file_name"]
        fsid  = int(row["fsID"])
        if cid == 6:
            class_map[fname] = ("gunshot", fsid)
        elif cid == 8:
            class_map[fname] = ("siren_unknown", fsid)
        else:
            class_map[fname] = ("background", fsid)

    all_wavs = collect_wavs(URBAN_DIR)
    fname_to_path = {os.path.basename(p): p for p in all_wavs}

    copied = {"gunshot": 0, "siren_unknown": 0, "background": 0}

    for fname, (class_name, fsid) in tqdm(class_map.items(), desc="UrbanSound8K"):
        src = fname_to_path.get(fname)
        if src is None:
            continue
        new_name = f"us8k_{fsid}_{fname}"
        copy_file(src, os.path.join(RAW_DIR, class_name), new_name)
        copied[class_name] += 1

    for k, v in copied.items():
        print(f"  {k}: {v} files copied")


# ---------------------------------------------------------------------------
# 2. ZENODO GUNSHOT DATASET
# ---------------------------------------------------------------------------

def process_zenodo():
    print("\n--- Zenodo Gunshot Dataset ---")

    if not os.path.isdir(ZENODO_DIR):
        raise FileNotFoundError(f"Zenodo folder not found: {ZENODO_DIR}")

    all_wavs   = collect_wavs(ZENODO_DIR)
    mono_files = [
        p for p in all_wavs
        if "_chan" not in os.path.basename(p).lower()
    ]

    print(f"  Total wav files found : {len(all_wavs)}")
    print(f"  Mono/mean files kept  : {len(mono_files)}")

    for src in tqdm(mono_files, desc="Zenodo"):
        new_name = "zenodo_" + os.path.basename(src)
        copy_file(src, os.path.join(RAW_DIR, "gunshot"), new_name)

    print(f"  gunshot: {len(mono_files)} files copied")


# ---------------------------------------------------------------------------
# 3. SIRENNET
# ---------------------------------------------------------------------------

SIRENNET_CLASS_MAP = {
    "ambulance": "ambulance_siren",
    "firetruck": "firetruck_siren",
    "police":    "police_siren",
    "traffic":   "background",
}

def process_sirennet():
    print("\n--- sireNNet ---")

    if not os.path.isdir(SIREN_DIR):
        raise FileNotFoundError(f"sireNNet folder not found: {SIREN_DIR}")

    for src_folder_name, target_class in SIRENNET_CLASS_MAP.items():
        src_folder = os.path.join(SIREN_DIR, src_folder_name)

        if not os.path.isdir(src_folder):
            print(f"  [WARNING] folder not found, skipping: {src_folder}")
            continue

        wavs = collect_wavs(src_folder)
        for src in tqdm(wavs, desc=f"sireNNet/{src_folder_name}"):
            new_name = f"sirennet_{src_folder_name}_" + os.path.basename(src)
            copy_file(src, os.path.join(RAW_DIR, target_class), new_name)

        print(f"  {target_class}: {len(wavs)} files copied")


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_summary():
    print("\n--- Final data/raw/ summary ---")
    total = 0
    for c in CLASSES:
        folder = os.path.join(RAW_DIR, c)
        n = len([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
        print(f"  {c:<20}: {n:>5} files")
        total += n
    print(f"  {'TOTAL':<20}: {total:>5} files")


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
        print("ERROR - the following folders were not found:")
        for m in missing:
            print(m)
        print("\nCheck the paths at the top of this file and try again.")
        exit(1)

    process_urbansound8k()
    process_zenodo()
    process_sirennet()
    print_summary()