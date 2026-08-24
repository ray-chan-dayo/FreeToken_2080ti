#!/usr/bin/env python3
"""fetch_marlin_sm75.py — Downloads the Marlin WNA16 sm_75 kernel sources from
weicj/vLLM-2080Ti-Definitive at build time and generates the sm_75 instantiation
files via generate_kernels.py.

Called automatically by setup.py when building the marlin_sm75 extension.
Can also be run standalone: python scripts/fetch_marlin_sm75.py

To update the pinned commit:
  1. Pick a new commit SHA from the upstream repo.
  2. Run this script once (it will fail checksum verification on new files).
  3. Update the SHA256 dict below with the values printed to stderr.
  4. Commit both this file and the updated SHA256 dict.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "weicj/vLLM-2080Ti-Definitive"

# Pinned to a specific commit for reproducibility and supply-chain safety.
# Do NOT change this to a branch name — branch HEADs are mutable.
# To update: see the docstring above.
COMMIT = "4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be"

BASE_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/csrc/moe/marlin_moe_wna16"
QUANTIZATION_BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/csrc/quantization/marlin"
SCALAR_TYPE_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/csrc/core/scalar_type.hpp"
REGISTRATION_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/csrc/core/registration.h"

# Destination: python/freetoken/kernel/csrc/marlin_wna16/
DEST = Path(__file__).parent.parent / "python" / "freetoken" / "kernel" / "csrc" / "marlin_wna16"

FILES = [
    "kernel.h",
    "marlin_template.h",
    "ops.cu",
    "generate_kernels.py",
]

# Additional headers that marlin_template.h transitively includes from vLLM's
# csrc/quantization/marlin/ tree. We vendor minimal stubs for the symbols we
# actually use at sm_75 (no FP8, no NVFP4).
QUANTIZATION_FILES = [
    "marlin.cuh",
    "marlin_dtypes.cuh",
    "marlin_mma.h",
    "dequant.h",
]

# SHA-256 of each file at the pinned commit above.
# Re-generate with: sha256sum <downloaded-file>
_SHA256 = {
    "kernel.h":          "47c243aaea4f95febee24f1c6abb25ffdc16d25521d32509af57156a411e9364",
    "marlin_template.h": "8f121a6820d46125040f13addb889038628bf9f8caac988838834723954675fc",
    "ops.cu":            "d7bdcf5a0d5124d3e8dcd6813aa8ef33479295e35b51b0457236d749b92ce4aa",
    "generate_kernels.py": "061b8b7a5ff25682fbc8db18ac2e0fe5b398223c5519c492e1b0d4563411eee4",
    "marlin.cuh":        "2bcc2d97933958c4da409e4d44ddaa62048b2372a85696f02346fc86dde1aaaa",
    "marlin_dtypes.cuh": "15b90a65eadb200a3f7165e2e0d7d899f36d4993b20dbd89093ef2cea274da71",
    "marlin_mma.h":      "bf863d252bfc468eaff42b2c1bda583c5e6ab97ceacf0ac248c164564ddcce9b",
    "dequant.h":         "39c4640d2de39374ef5d96e7fa27701a675922113c3f5b05d45369321090e728",
    "scalar_type.hpp":   "bc4800ca8e0bf65e455e2a747b1c6de515a79b870ae4788a3d966988597af495",
    "registration.h":    "58b98cd4792f18baae3d316af7a91440ec991ee75d2641d2bc805bcc8cd1c02e",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fname = dest.name
    expected = _SHA256.get(fname)

    if dest.exists():
        if expected and _sha256(dest) == expected:
            print(f"  [skip]  {fname} (already exists, checksum ok)")
            return
        # Present but checksum doesn't match — re-fetch (could be a partial download
        # or a stale file from a previous pinned commit).
        print(f"  [refetch] {fname} (checksum mismatch, re-downloading)")
        dest.unlink()

    print(f"  [fetch] {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    if expected:
        got = _sha256(dest)
        if got != expected:
            dest.unlink()
            raise RuntimeError(
                f"SHA-256 mismatch for {fname}:\n"
                f"  expected {expected}\n"
                f"  got      {got}\n"
                f"The upstream file changed at commit {COMMIT}. "
                "Update the _SHA256 dict in this script if you intentionally changed the pin."
            )


def generate_kernels(dest: Path, arch: str = "7.5") -> None:
    """Run generate_kernels.py to produce sm75_kernel_*.cu and kernel_selector.h."""
    script = dest / "generate_kernels.py"
    if not script.exists():
        raise FileNotFoundError(f"generate_kernels.py not found at {script}")
    print(f"  [gen]   generate_kernels.py for arch {arch}")
    result = subprocess.run(
        [sys.executable, str(script), arch],
        cwd=str(dest),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"generate_kernels.py failed:\n{result.stdout}\n{result.stderr}"
        )
    generated = sorted(dest.glob("sm75_kernel_*.cu"))
    print(f"  [gen]   generated {len(generated)} sm75 kernel files")
    selector = dest / "kernel_selector.h"
    if not selector.exists():
        raise FileNotFoundError("kernel_selector.h was not generated")


def main() -> None:
    print(f"Fetching Marlin WNA16 sm_75 sources into {DEST} ...")
    DEST.mkdir(parents=True, exist_ok=True)

    # Core Marlin files
    for fname in FILES:
        fetch(f"{BASE_URL}/{fname}", DEST / fname)

    # Transitive includes: quantization/marlin/
    quant_dir = DEST / "quantization" / "marlin"
    for fname in QUANTIZATION_FILES:
        fetch(f"{QUANTIZATION_BASE}/{fname}", quant_dir / fname)

    # core/scalar_type.hpp and core/registration.h
    core_dir = DEST / "core"
    fetch(SCALAR_TYPE_URL, core_dir / "scalar_type.hpp")
    fetch(REGISTRATION_URL, core_dir / "registration.h")

    # Generate sm75_kernel_*.cu + kernel_selector.h
    generate_kernels(DEST, arch="7.5")

    print("Done. Marlin sm_75 sources ready.")


if __name__ == "__main__":
    main()
