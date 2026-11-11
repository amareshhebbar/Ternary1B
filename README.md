# Ternary1B

Take `HuggingFaceTB/SmolLM2-1.7B`, which stores every weight in 16 bits (about 3.4 GB), and squeeze its attention and feed forward weights down to 2 bits or 1.58 bit ternary (minus 1, 0 or plus 1). Six models compare six ways to survive that squeeze, on one table against the original model and plain rounding.

Every model is pushed to the Hugging Face Hub, added to the Ternary1B collection, downloaded back from the Hub and tested before the next model starts. No external APIs are used.

## What you get

<table>
<tr><th>Model</th><th>Bits</th><th>How it survives</th><th>Training</th></tr>
<tr><td>GPTQ</td><td>2</td><td>Rounds layer by layer and corrects each rounding error using the layer inputs</td><td>None, about 15 minutes</td></tr>
<tr><td>AWQ</td><td>2</td><td>Protects weights connected to large activations before rounding</td><td>None, about 10 minutes</td></tr>
<tr><td>GPTQ_LORA</td><td>2 plus adapter</td><td>The GPTQ model, frozen, plus a small adapter trained to repair the damage</td><td>Distillation, short</td></tr>
<tr><td>HQQ_LORA</td><td>2 plus adapter</td><td>Fast calibration free HQQ rounding plus a repair adapter</td><td>Distillation, short</td></tr>
<tr><td>QAT</td><td>2</td><td>The whole model trains while rounded, so it learns weights that survive rounding</td><td>Distillation, long</td></tr>
<tr><td>BITNET</td><td>1.58</td><td>BitNet b1.58 layers (ternary weights, 8 bit activations), eased in by a warmup</td><td>Distillation, long</td></tr>
</table>

Baseline rows: the original bf16 model, plain 2 bit rounding, and plain ternary rounding.

All training uses knowledge distillation: the original model is the teacher and the squeezed model learns to copy its predictions.

Tests: WikiText2 perplexity, HellaSwag, ARC Easy, PIQA and MMLU (zero shot), packed file size, and CPU speed with llama.cpp for the ternary model.

## How the files are stored

Each model repo on the Hub holds:

* the weights expanded to bf16 safetensors in the root, so anyone can load it in transformers with no extra code,
* `packed/`: the real compressed model (4 weights per byte at 2 bits, 5 weights per byte for ternary, plus scales). Its size is the size reported in the table,
* `tq_runtime.py`: rebuilds the model from `packed/`, and adds 8 bit activations for BitNet.

The verify step checks that `packed/` rebuilds the same weights as the root copy.

## Step 1: push to GitHub (on your machine)

Create an empty GitHub repo named `Ternary1B` (no README), then:

```bash
unzip Ternary1B.zip && cd Ternary1B
git init
git branch -M main
git add .
git commit -m "Ternary1B pipeline"
git remote add origin https://github.com/amareshhebbar/Ternary1B.git
git push -u origin main
```

Create a Hugging Face token with Write access at huggingface.co/settings/tokens.

## Step 2: start the pod

* GPU: one 48 GB card. L40S is the best fit; A6000, A100 and H100 also work. QAT and BitNet train the whole model in fp32 alongside a frozen teacher, so 24 GB cards are not enough without lowering the batch.
* Volume: at least 100 GB mounted at `/workspace` (network volume, so a restart keeps progress).
* Template: any PyTorch template with CUDA 12.

## Step 3: install and check

```bash
cd /workspace
git clone https://github.com/amareshhebbar/Ternary1B.git && cd Ternary1B
bash setup.sh
export HF_TOKEN=hf_yourtokenhere
export HF_HOME=/workspace/hf_cache
source .venv/bin/activate
python run.py check
```

Do not export `TQ_WORK`; each config has its own run folder.

<table>
<tr><th>Check</th><th>What it proves</th></tr>
<tr><td>sys</td><td>Libraries import, CUDA and bf16 work, VRAM and disk are enough</td></tr>
<tr><td>hub</td><td>Token has write access: creates a private repo and a collection, then deletes both</td></tr>
<tr><td>data</td><td>Base model and tokenizer load, layer sizes fit the group size, FineWeb Edu streams, WikiText2 loads</td></tr>
<tr><td>quant</td><td>Packing round trips exactly; GPTQ and AWQ beat plain rounding on a test layer; packed files rebuild the weights</td></tr>
<tr><td>train</td><td>Real QAT, BitNet and LoRA repair steps with the teacher fit in memory; prints tokens per second and an estimate of total training hours</td></tr>
<tr><td>cpu</td><td>Tools needed to build llama.cpp are present</td></tr>
</table>

Read the time estimate from the train check before starting the full run. Run one part alone with `python run.py check train`.

## Step 4: run

```bash
tmux new -s tq
source .venv/bin/activate
export HF_TOKEN=hf_yourtokenhere
export HF_HOME=/workspace/hf_cache
python run.py pipeline
```

Detach with `Ctrl+b` then `d`, reattach with `tmux attach -t tq`. Follow progress with `tail -f /workspace/ternary1b_runs/logs/*.log` in another terminal.

Steps, in order:

1. `data`: tokenizes FineWeb Edu into calibration samples and training chunks (calibration documents are kept out of training), and caches WikiText2.
2. `dataset_push`: pushes the token data to the Hub and checks row counts.
3. `baseline`: evaluates the original model, plain 2 bit rounding and plain ternary rounding.
4. For GPTQ, AWQ, GPTQ_LORA, HQQ_LORA, QAT and BITNET:
   1. `train`: quantizes (and trains where the method needs it), then writes the bf16 copy and `packed/`
   2. `push`: uploads everything plus the model card and adds it to the collection
   3. `verify`: downloads from the Hub, checks every file, rebuilds weights from `packed/`, runs a quick test, and stops the pipeline if the model is broken
   4. `evalgen`: the full test suite
   5. `publish`: updates the card with the results table and charts
5. `cpubench`: builds llama.cpp on the CPU, converts the original model (f16 and 4 bit Q4_K_M) and the BitNet model (TQ2_0 and TQ1_0), and measures tokens per second. If anything here fails, it records not available and the pipeline continues.
6. `report`: final charts and cards for every model.

If the pod stops, run `python run.py pipeline` again. Finished steps are skipped.

## Two configs

<table>
<tr><th></th><th>configs/config.yaml (default)</th><th>configs/config_big_budget.yaml</th></tr>
<tr><td>QAT and BitNet training</td><td>1,800 steps each, about 59M tokens</td><td>8,000 steps each, about 262M tokens</td></tr>
<tr><td>LoRA repair</td><td>300 steps, about 10M tokens</td><td>1,500 steps, about 49M tokens</td></tr>
<tr><td>Calibration</td><td>128 x 2,048 tokens</td><td>256 x 2,048 tokens</td></tr>
<tr><td>Benchmarks</td><td>1,000 examples per task</td><td>Full tasks</td></tr>
<tr><td>Run folder</td><td>/workspace/ternary1b_runs</td><td>/workspace/ternary1b_runs_big</td></tr>
<tr><td>Rough time on an L40S</td><td>about 11 to 15 hours</td><td>about 40 to 55 hours</td></tr>
</table>

Times are estimates; the train check prints a real one. To run the big config, prefix every command:

```bash
TQ_CONFIG=configs/config_big_budget.yaml python run.py check
TQ_CONFIG=configs/config_big_budget.yaml python run.py pipeline
```

Both configs push to the same Hub repos, so a big run upgrades the models in place.

## Useful commands

```bash
python run.py status
python run.py steps
python run.py run bitnet:verify
python run.py pipeline from=qat:train to=qat:publish
python run.py reset bitnet:train
python run.py reset all
```

## Saving money

* GPTQ and AWQ take minutes; QAT and BitNet are where the money goes. To trim, lower `qat.steps` and `bitnet.steps`.
* Stop the pod as soon as `pipeline complete` prints.
* A spot or interruptible pod works because the pipeline resumes.
* If budget runs out midway, every finished model is already on the Hub with its card.

## If something fails

* Out of memory in the train check: set `train.micro_batch: 4` and `train.grad_accum: 8` (same tokens per step, less memory).
* `token belongs to X but config hf_user is Y`: run `export HF_USER=X`.
* Verify failed: see `/workspace/ternary1b_runs/results/<model>_verify.json`. A perplexity gate failure means the model collapsed; rerun with `python run.py reset <model>:train` and a lower learning rate.
* CPU benchmark shows not available: install build tools with `apt-get install -y build-essential git cmake`, then `python run.py reset cpubench` and `python run.py run cpubench`.
* Any other failure: the full log is in `/workspace/ternary1b_runs/logs/<step>.log`.

## What to expect

BitNet models trained ternary from scratch use trillions of tokens. Converting an existing model with millions of tokens recovers only part of the original quality, so expect a clear gap to the bf16 row, largest on MMLU. The point of the table is which method keeps the most quality per megabyte, and how far training moves each squeezed model away from plain rounding.

## Hub names

<table>
<tr><td>Dataset</td><td>AmareshHebbar/Ternary1B_Data_SmolLM2_1.7B</td></tr>
<tr><td>Models</td><td>AmareshHebbar/Ternary1B_SmolLM2_1.7B_GPTQ, _AWQ, _GPTQ_LORA, _HQQ_LORA, _QAT, _BITNET</td></tr>
<tr><td>Collection</td><td>Ternary1B</td></tr>
</table>

## Layout

<table>
<tr><td>run.py</td><td>Entry point: check, pipeline, run, status, reset</td></tr>
<tr><td>tq/quant.py</td><td>Rounding, GPTQ, AWQ, HQQ, ternary, packing and unpacking</td></tr>
<tr><td>tq/ptq.py</td><td>Layer by layer calibration pass for GPTQ and AWQ</td></tr>
<tr><td>tq/layers.py</td><td>QAT and BitLinear layers with the straight through estimator</td></tr>
<tr><td>tq/train.py</td><td>Distillation training for LoRA repair, QAT and BitNet</td></tr>
<tr><td>tq/export.py</td><td>Writes the bf16 copy, packed files and runtime</td></tr>
<tr><td>tq/data.py, tq/dataset.py</td><td>FineWeb Edu and WikiText2, dataset push</td></tr>
<tr><td>tq/evaluate.py</td><td>Perplexity, benchmarks, sample completions</td></tr>
<tr><td>tq/cpubench.py, scripts/build_llamacpp.sh</td><td>llama.cpp build, GGUF conversion and CPU timing</td></tr>
<tr><td>tq/verify.py, tq/hub.py, tq/card.py, tq/report.py</td><td>Hub checks, uploads, model cards, charts</td></tr>
<tr><td>tq/check.py</td><td>Environment checks</td></tr>
</table>
