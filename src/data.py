"""
Data: TinyStories, read as raw bytes.

* Train: the first ~200 MB of TinyStoriesV2-GPT4-train.txt (an HTTP range request,
  so we never download the full 2.2 GB). 200 MB of bytes is about 2x what the
  longest run consumes, so no run ever sees the same text twice.
* Validation: the full TinyStoriesV2-GPT4-valid.txt (22.5 MB), a separate file.
* The story separator "<|endoftext|>" (13 bytes) is replaced by a single byte 0.

Every run sees EXACTLY the same batch at the same step number: the batch for step s
is drawn from a random generator seeded with (seed, s). So when we compare the dense
run and the MoE run at step 4000, they were fed identical text. It also means a run
that resumes after a Colab disconnect gets the same batches it would have got.
"""
import os
import urllib.request

import numpy as np
import torch

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
VALID_FILE = "TinyStoriesV2-GPT4-valid.txt"
EOT = b"<|endoftext|>"


def _download(url, dest, max_bytes=None):
    req = urllib.request.Request(url)
    if max_bytes is not None:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _to_bytes_array(raw: bytes) -> np.ndarray:
    # cut at the last complete story so a range download doesn't end mid-story
    last = raw.rfind(EOT)
    if last > 0:
        raw = raw[: last + len(EOT)]
    raw = raw.replace(EOT, b"\x00")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def prepare_data(data_dir, train_mb=200):
    """Downloads (once) and converts to .bin files of uint8. Returns their paths."""
    os.makedirs(data_dir, exist_ok=True)
    out_train = os.path.join(data_dir, "train.bin")
    out_val = os.path.join(data_dir, "val.bin")
    if os.path.exists(out_train) and os.path.exists(out_val):
        return out_train, out_val
    for name, out, mb in [(TRAIN_FILE, out_train, train_mb), (VALID_FILE, out_val, None)]:
        tmp = out + ".download"
        print(f"downloading {name} ({'first %d MB' % mb if mb else 'full file'}) ...")
        _download(BASE + name, tmp, None if mb is None else mb * 1024 * 1024)
        with open(tmp, "rb") as f:
            arr = _to_bytes_array(f.read())
        arr.tofile(out)
        os.remove(tmp)
        print(f"  -> {out}: {len(arr):,} bytes")
    return out_train, out_val


def write_synthetic_data(data_dir, n_train=2_000_000, n_val=200_000, seed=0):
    """Offline fallback used ONLY by the CPU smoke test and unit tests: a little
    procedurally generated story-like text. Never used for the reported results."""
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = ["Lily", "Tom", "Ben", "Mia", "Sam", "Anna"]
    things = ["ball", "cat", "tree", "cake", "boat", "kite", "dog", "box"]
    feels = ["happy", "sad", "scared", "proud", "tired"]
    def story():
        n, t, f = rng.choice(names), rng.choice(things), rng.choice(feels)
        return (f"Once upon a time, {n} had a {t}. {n} was very {f}. "
                f"One day the {t} was lost. {n} looked and looked. "
                f"At last {n} found the {t} and was {rng.choice(feels)} again.\x00")
    for out, n in [("train.bin", n_train), ("val.bin", n_val)]:
        buf = bytearray()
        while len(buf) < n:
            buf += story().encode()
        np.frombuffer(bytes(buf[:n]), dtype=np.uint8).tofile(os.path.join(data_dir, out))
    return os.path.join(data_dir, "train.bin"), os.path.join(data_dir, "val.bin")


class ByteData:
    def __init__(self, train_path, val_path, block_size, batch_size, seed, device):
        self.train = np.memmap(train_path, dtype=np.uint8, mode="r")
        self.val = np.memmap(val_path, dtype=np.uint8, mode="r")
        self.T, self.B, self.seed, self.device = block_size, batch_size, seed, device

    def _batch(self, arr, rng):
        ix = rng.integers(0, len(arr) - self.T - 1, size=self.B)
        x = np.stack([arr[i:i + self.T] for i in ix]).astype(np.int64)
        y = np.stack([arr[i + 1:i + 1 + self.T] for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y

    def train_batch(self, step):
        # same step number -> same batch, in every run
        return self._batch(self.train, np.random.default_rng([self.seed, step]))

    def val_batches(self, n):
        # a fixed set of validation batches, identical for every evaluation in every run
        rng = np.random.default_rng([self.seed, 10_000_019])
        return [self._batch(self.val, rng) for _ in range(n)]

    def tokens_per_step(self):
        return self.B * self.T
