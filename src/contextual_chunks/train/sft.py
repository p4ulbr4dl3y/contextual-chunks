"""
Text SFT (contextual retrieval distillation) for Qwen3.5-0.8B on top of the
teacher-labelled contextual-chunks dataset.

TEXT-ONLY task. The original Unsloth Qwen3.5 notebook trains a VISION model on
LaTeX-OCR images; here there are no images -> FastLanguageModel, plain chat
template, plain text collator.

Reference implementation / smoke test. Run on a GPU (Colab T4) after unsloth
is installed (see train_contextual_sft.ipynb). Requires trl==0.22.2 like the
original notebook.

    .venv/bin/python -m contextual_chunks.train.sft
    .venv/bin/python -m contextual_chunks.train.sft --data local --max-steps 20
    .venv/bin/python -m contextual_chunks.train.sft --model unsloth/Qwen3-0.6B --no-grad-ckpt   # fast smoke test

NOTE: the currently published dataset has ~73% of rows with the chunk OUTSIDE
the document shown in the prompt (legacy truncation bug). Absolute quality is
meaningless -- treat this as a PIPELINE SMOKE TEST. Rebuild data with
build_chunks.py + re-label before a real run.

SPEED (measured on a Colab Tesla T4, 96-sample subset, bs2 x ga4):

    model / config                      dtype   ms/1k tok   s / sample
    unsloth/Qwen3.5-0.8B (reference)    fp32      1.50         3.63    <- forced fp32
    unsloth/Qwen3-0.6B                  fp16      0.43         2.05    <- -1.8x end-to-end
    unsloth/Qwen3-0.6B, no grad-ckpt    fp16      0.34         1.39    (OOMs at bs>=8)

Why Qwen3.5-0.8B is stuck at fp32 on a T4:
  * unsloth lists `qwen3_5` in FORCE_FLOAT32 (GDN layers overflow to NaN in fp16)
  * a T4 is Turing (sm_75) and has NO bf16 hardware (Bfloat16 = FALSE)
  * unsloth/models/loader.py forces float32 whenever `not SUPPORTS_BFLOAT16`, so
    no dtype= argument can avoid the fallback
  * nothing fixes the numerics: forcing fp16 gives `grad_norm = nan` on 5 of the
    first 6 steps and buys ~0% throughput (unsloth already autocasts the matmuls)
  * faking bf16 (`torch.cuda.is_bf16_supported = lambda: True`) only satisfies the
    HF/Unsloth validators -- sm_75 still has no bf16 kernels -> no speedup
  * flash-linear-attention (`fla`) does not help either (28.5 vs 28.9 s/step)
  * bs4/ga2 = 32.6 s/step, bs8/ga1 = 43.1 s/step (+ host-RAM OOM), no grad-ckpt
    = 28.8, `group_by_length` is ignored -> training is compute-bound in fp32

=> the only real lever is a TEXT-ONLY student: `--model unsloth/Qwen3-0.6B` trains in
   fp16 with zero NaN, ~1.8x faster end-to-end (111 min -> ~60 min). A ~15 min run on
   a T4 is NOT reachable -- that needs bf16 hardware (L4/A100) or fewer/short epochs.
"""
import argparse
import glob
import os
import re

from contextual_chunks import paths

from datasets import load_dataset

try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTTrainer, SFTConfig
except ImportError as exc:  # unsloth ships Linux wheels only, the run happened on Colab
    raise SystemExit(
        f"the training stack is not installed ({exc}).\n"
        "unsloth ships Linux wheels only, so the SFT runs on a CUDA host:\n"
        "  uv pip install -e '.[train]'"
    ) from exc

MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen3.5-0.8B")
# Fast alternative (text-only, fp16 works on a T4 -> ~1.8x end-to-end, NaN-free):
#   --model unsloth/Qwen3-0.6B
HF_DATA = "p4ulbr4dl3y/ru-contextual-chunks"
LOCAL_DATA = paths.SFT

MAX_SEQ_LENGTH = 16384        # window (<=14000 chars ~ 11k tok) + chunk + answer
# Where trainer checkpoints land. On Colab point this at Drive so a disconnect
# does not kill the run, e.g.
#   --out /content/drive/MyDrive/contextual_chunks/qwen_contextual_sft
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs_contextual_sft")
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def load_raw(data_source):
    if data_source == "local":
        ds = load_dataset("json", data_files=LOCAL_DATA, split="train")
    else:
        ds = load_dataset(HF_DATA, split="train")

    # window_hash groups identical prompts (near-duplicate documents share
    # whole windows); doc_hash is the fallback, `id` the last resort.
    return ds.map(lambda ex: {"doc_id": (ex.get("window_hash") or ex.get("doc_hash")
                                        or re.sub(r"_chunk_\d+$", "", ex["id"]))})


def group_split(ds, val_frac=0.15, seed=3407):
    """Hold out WHOLE documents, never random chunks (chunks of one doc are
    near-duplicates -> random split leaks)."""
    doc_ids = sorted(set(ds["doc_id"]))
    n_val = max(1, int(len(doc_ids) * val_frac))
    import random
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    val_docs = set(doc_ids[:n_val])
    train = ds.filter(lambda ex: ex["doc_id"] not in val_docs)
    val = ds.filter(lambda ex: ex["doc_id"] in val_docs)
    print(f"[split] docs={len(doc_ids)} train={len(train)} val={len(val)} (val docs={n_val})")
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["hf", "local"], default="hf")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--model", default=MODEL_NAME,
                    help="base model; use unsloth/Qwen3-0.6B on a T4 for fp16 speed")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (~1.3x faster, more VRAM)")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", default=OUTPUT_DIR,
                    help="checkpoint dir; point at Drive on Colab")
    a = ap.parse_args()
    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)

    # ---- model -------------------------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        a.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # ---- tokenizer / chat-template sanity checks ---------------------------
    print("[tok]", type(tokenizer).__name__)
    print("[tok] eos:", repr(tokenizer.eos_token), tokenizer.eos_token_id)
    print("[tok] pad:", repr(tokenizer.pad_token), tokenizer.pad_token_id)
    # Qwen3.5 ships a *processor* (Qwen3VLProcessor) whose __call__ treats the
    # first positional argument as IMAGES. Two consequences:
    #   1. always tokenize text with an explicit keyword:  tokenizer(text=...)
    #   2. the default pad token is <|vision_pad|> -> use eos for text-only SFT.
    tokenizer.pad_token = tokenizer.eos_token
    # The template injects an empty  thinking block into the assistant turn
    # (enable_thinking is ignored). Train and infer with the SAME template.

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,            # 0 enables unsloth fast patching (no reg needed here)
        bias="none",
        use_gradient_checkpointing=("unsloth" if not a.no_grad_ckpt else False),
        random_state=a.seed,
    )

    # ---- data --------------------------------------------------------------
    ds = load_raw(a.data)
    train_ds, val_ds = group_split(ds, seed=a.seed)

    def to_text(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}

    train_ds = train_ds.map(to_text)
    val_ds = val_ds.map(to_text)

    # NOTE: tokenizer(text=...) -- positional arg is reserved for images.
    lengths = [len(ids) for ids in
               tokenizer(text=list(train_ds["text"][:64]), add_special_tokens=False)["input_ids"]]
    lengths.sort()
    print(f"[len] sample tokens min/med/max: "
          f"{lengths[0]}/{lengths[len(lengths)//2]}/{lengths[-1]}")
    if lengths[-1] > MAX_SEQ_LENGTH:
        print("WARNING: samples exceed max_seq_length and will be truncated!")

    # ---- train (trl 0.22.x API) -------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            packing=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            # CRITICAL: HF defaults per_device_eval_batch_size to 8. Unsloth's eval path
            # forces UNSLOTH_RETURN_LOGITS=1 (ignoring prediction_loss_only), so one eval
            # batch materialises [batch x seq x 151936] logits -> 12.45 GiB -> CUDA OOM,
            # and it happens BEFORE the first save, so the whole run is lost.
            # Measured peak on a T4: bs=8 OOM | bs=2 14.39 GB | bs=1 7.91 GB.
            per_device_eval_batch_size=1,
            prediction_loss_only=True,
            eval_accumulation_steps=1,
            warmup_steps=10,
            max_steps=a.max_steps,
            num_train_epochs=a.epochs if a.max_steps < 0 else 1,
            learning_rate=a.lr,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=a.seed,
            output_dir=out_dir,
            report_to="none",
            eval_strategy="steps",
            eval_steps=20,
            save_strategy="steps",
            save_steps=20,          # must be a multiple of eval_steps (HF validation)
            save_total_limit=3,
            save_only_model=False,      # keep optimizer state -> real resume
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
    )

    # train ONLY on the assistant context, never on the document prompt
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )

    from transformers import EarlyStoppingCallback
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # resume automatically if the checkpoint dir already has checkpoints
    ckpts = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    resume = ckpts[-1] if ckpts else None
    print("[ckpt] resume from:", resume or "none (fresh start)")

    stats = trainer.train(resume_from_checkpoint=resume)
    print(stats.metrics)

    final = os.path.join(out_dir, "final_lora")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print("saved LoRA ->", final)


if __name__ == "__main__":
    main()
