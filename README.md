# entropy_wall

A tiny LavaRand-style entropy tool. Point it at a photo, a video file, or
your webcam, and it hashes the pixel data into a random seed — the same
basic idea as Cloudflare's [Wall of Entropy](https://blog.cloudflare.com/randomness-101-lavarand-in-production/):
they point cameras at a wall of lava lamps; this points a camera at
whatever you've got.

## How it works

1. **Absorb** — every frame's pixel bytes get folded into a running
   SHA3-512 hash, along with a domain-separation tag, a frame counter, and
   a high-resolution timestamp. This lives in [`entropy/pool.py`](entropy/pool.py).
2. **Derive** — the accumulated hash is never handed out directly. It's run
   through HKDF (RFC 5869: HMAC-based Extract-and-Expand, using a random
   local salt) to whiten it and expand it to as many output bytes as you
   ask for.

Optionally (`--noise-extract`), instead of hashing raw frames it hashes
either the **frame-to-frame difference** (video/webcam — cancels out the
static scene, leaving mostly sensor noise and motion) or the **least
significant bit plane** (a single photo — the low bits of a photo are
dominated by sensor noise and compression artifacts, not the actual image).
This weights the pool toward the parts of the signal most likely to be
physically unpredictable.

## ⚠️ Important caveat

HKDF will make the *output* look statistically uniform even if the *input*
had almost no real randomness (a photo of a blank wall, a frozen video).
**A hash that looks random is not the same as an unpredictable secret** —
unpredictability has to come from the source, not the hash function. Use
the printed "source entropy" diagnostic as a sanity check, and prefer live,
changing, physically noisy input (a webcam on something with motion) over
a single static photo.

This tool is for learning/experimenting with the LavaRand idea. For
anything security-sensitive, use your OS's CSPRNG directly
(Python's `secrets` / `os.urandom`) — that's what Cloudflare's actual
production randomness relies on too; the lava lamp wall is one of several
entropy sources mixed in, not a replacement for a real CSPRNG.

## Install

```bash
pip install -r requirements.txt
```

## GUI

There's also a Tkinter GUI (`Tkinter` ships with Python, so no extra
dependency beyond what's in `requirements.txt`):

```bash
python gui.py
```

Tabs for Photo / Video / Webcam, the same seed-length and `--noise-extract`
options as the CLI, a live webcam preview, and a Stream mode that keeps
emitting fresh seeds on an interval (mirrors `webcam --stream`). Capture and
hashing run on a background thread so the window doesn't freeze. Copy the
derived seed to the clipboard or save it to a file from the Result panel.

## CLI usage

```bash
# Hash a single photo into a 32-byte seed
python entropy_wall.py image photo.jpg

# Hash every 3rd frame of a video into a 64-byte seed
python entropy_wall.py video clip.mp4 --sample-every 3 --length 64

# Capture 5 seconds from your webcam
python entropy_wall.py webcam --seconds 5

# Weight toward sensor noise instead of raw pixels
python entropy_wall.py image photo.jpg --noise-extract

# Save the raw seed bytes to a file instead of just printing hex
python entropy_wall.py image photo.jpg --out seed.bin

# Continuous mode: keep capturing forever, emit a fresh seed every 2s
# (Ctrl+C to stop) — closest to how the real Wall of Entropy runs
python entropy_wall.py webcam --stream --interval 2
```

Each command prints diagnostics (frame count, estimated source entropy) to
stderr and the derived seed (hex) to stdout, so you can pipe just the seed:

```bash
python entropy_wall.py image photo.jpg 2>/dev/null > seed.hex
```

## Using the seed

The output is raw seed material, not a random number generator by itself.
[`examples/use_seed.py`](examples/use_seed.py) shows how to turn it into an
actual stream of numbers using both stdlib `random.Random` and
`numpy.random.Generator(PCG64(...))`:

```bash
python entropy_wall.py image photo.jpg 2>/dev/null | python examples/use_seed.py -

# or from a saved seed file
python entropy_wall.py image photo.jpg --out seed.bin
python examples/use_seed.py seed.bin
```

Both generators above are deterministic and **not cryptographically
secure** — fine for simulations, games, procedural generation, anything
where reproducibility matters more than unpredictability. If you need
actual cryptographic randomness, use `secrets`/`os.urandom` directly
instead of seeding anything with this tool's output.

## Project layout

```
entropy_wall.py        CLI entry point
gui.py                  Tkinter GUI front end
entropy/
  pool.py               EntropyPool: absorb (SHA3-512) + derive (HKDF)
  capture.py            frame sources: image / video file / webcam, + noise extraction
  pipeline.py            shared "absorb frames into a pool" helper (used by CLI + GUI)
  diagnostics.py        Shannon entropy estimator (sanity check only)
examples/
  use_seed.py            turns a derived seed into actual random numbers (random.Random / numpy PCG64)
```
