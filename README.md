# blur-1

A class-conditional DDPM trained from scratch on a MacBook.

23M parameters, trained end-to-end on an M2 Pro in about **3.7 hours** on **5,202
paintings** by Van Gogh, Renoir, Monet and Cézanne pulled from WikiArt. No
pretrained weights, no latent space, no cloud — the UNet, the noise schedule, the
training loop and the sampler are all in the notebook.

Conditioned on **two independent axes**, so you can ask for a specific *kind* of
image rather than just sampling the whole distribution:

| axis | values |
|---|---|
| **artist** | `van_gogh`, `renoir`, `monet`, `cezanne` |
| **genre** | `landscape`, `portrait`, `still_life`, `cityscape`, `genre_painting`, `other` |

```python
sample(artist="van_gogh", genre="landscape", w_artist=3.0, w_genre=2.0)
```

Each axis gets its own classifier-free-guidance strength, and artists can be
blended (`0.5 monet + 0.5 van_gogh`) at sampling time.

## Contents

| path | what it is |
|---|---|
| `painting_diffusion.ipynb` | the model — dataset build, UNet, training loop, sampler |
| `painting_diffusion_testrun.ipynb` | a short run with outputs kept inline |
| `samples_test/step2000/` | samples from the 2,000-step test run |

## Running it

Run every cell top to bottom.

Section 2 scans the 72 WikiArt parquet shards, downloads only the **16** that
hold these four painters (~7.3 GB instead of 34 GB), writes each painting out as
a short-side-144 JPEG and deletes the shard, so peak disk stays around 500 MB.
It caches to `data/` and every later run skips it. Section 8 (training) is
checkpointed every 2,000 steps, so you can interrupt the kernel and re-run the
cell to resume.

Requires `torch` (MPS build), `numpy`, `pillow`, `matplotlib`, `tqdm`,
`pyarrow`, `huggingface_hub`, `torchvision`.

Nothing large is tracked in git — `data/` and the checkpoints are rebuilt
locally.

## Setup

| | |
|---|---|
| resolution | 64×64 (cached at 80×80, random-cropped down) |
| UNet | base 64 ch, mults `(1,2,3,4)`, 2 res blocks/level, attention at 16 and 8, 4 heads, dropout 0.1 |
| diffusion | 1,000 timesteps, cosine schedule (Nichol & Dhariwal), **v**-prediction |
| training | batch 16, lr 2e-4, 1,000-step warmup, 24,000 steps, grad clip 1.0 |
| EMA | decay 0.9995, with warmup; sampling always uses the EMA weights |
| sampling | DDIM, 50 steps, `eta=0` |

Most of the sizing is a 16 GB unified-memory constraint rather than a quality
choice. Width at the full 64×64 level drives activation memory, so it stays
narrow (64) and the capacity goes into the deeper, cheaper levels. Batch 16 is
also a memory choice — throughput is nearly flat from 16 to 32, but batch 64
crosses a cliff where MPS starts swapping and a step goes from 0.5 s to 54 s.

Two details that matter more than they look:

- **EMA warmup.** The shadow weights start at the random init, so a flat 0.9995
  decay leaves the average 74% initialisation even 600 steps in. Early previews
  come out as pure noise while the raw model is already painting — it looks
  exactly like a broken sampler.
- **Zero-init output layers.** Every residual block and the final conv start at
  zero, so the network begins as an identity map and early training is stable.

## Two-axis guidance

Ordinary CFG has one conditional and one unconditional prediction. To steer
artist and genre independently, this uses three and composes them:

```
v̂ = v_∅ + w_a · (v_a − v_∅) + w_g · (v_a,g − v_a)
```

The first term pulls toward the artist, the second adds the genre *on top of*
that artist. All three run as one batched forward pass, and `w_a == w_g`
recovers ordinary joint guidance. Low strength (1–2) is diverse but muddy, high
(5+) is vivid and over-saturated; 3 is a reasonable start.

## Next

- **Higher resolution.** `IMAGE_SIZE = 128`, `CACHE_SIZE = 144`, `BATCH_SIZE = 4`,
  then delete `data/cache_images.npy` and re-run section 3b. The JPEG archive is
  already stored at short side 144, so nothing needs re-downloading, and
  warm-starting from the 64px checkpoint converges much faster than starting
  over.
- **More painters.** Add WikiArt label ids to `ARTISTS` and delete
  `data/manifest.json`. Check the shard cost first — artists smeared across many
  shards get expensive fast.
- **Watch for memorisation.** With ~5k images this model can start copying.
  Nearest-neighbour a sample you like against the training cache to check.
