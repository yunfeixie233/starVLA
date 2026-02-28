# 3-Way Feature Decomposition (imgtext / img / txt) -- Modification Log

Created: 2026-02-27

## Goal

Decompose `feats_A` (VLM condition embedding from `hidden_states[-1]` masked mean-pool) into three variants:

1. **feats_A.pt** -- image+text pooling (backward-compatible, unchanged)
2. **feats_A_img.pt** -- pool over image tokens only
3. **feats_A_txt.pt** -- pool over task instruction tokens only (not template/system/special tokens)

Run at N=5K, compute CKNNA_proprio and CKNNA_action for each variant, generate CSV and plots.

---

## Files Modified

### Extraction scripts (7 scripts)

| Script | Location | Model(s) |
|--------|----------|----------|
| `extract_features_starvla.py` | `starVLA/cknna/` | 6 StarVLA models |
| `extract_features_openvla.py` | `SimplerEnv-OpenVLA/cknna/` | openvla-7b-bridge, openvla-7b-bridge-ft-200k |
| `extract_features_cogact.py` | `SimplerEnv-OpenVLA/cknna/` | CogACT-Small/Base/Large |
| `extract_features_spatialvla.py` | `SimplerEnv-OpenVLA/cknna/` | spatialvla-sft-bridge |
| `extract_features_pi0_lerobot.py` | `SimplerEnv-OpenVLA/cknna/` | pi0-lerobot-bridge |
| `extract_features_groot_n15.py` | `SimplerEnv-OpenVLA/cknna/` | groot-n15-bridge |
| `extract_features_groot_n16.py` | `SimplerEnv-OpenVLA/cknna/` | groot-n16-bridge |

### New files

| File | Purpose |
|------|---------|
| `starVLA/cknna/run_3way_5k.sh` | Orchestration script (all models + CKNNA) |
| `starVLA/cknna/record/scripts/generate_3way_csv.py` | Merge JSON results into CSV |
| `starVLA/cknna/record/scripts/plot_3way_cknna.py` | 3-panel scatter + bar comparison plots |

### Unchanged

- `compute_cknna_large.py` -- no change needed; just pass different `--feats_A` paths

---

## Common Changes Per Script

Each extraction script got:

1. **`find_subsequence(seq, subseq)`** -- O(n*m) substring search on token ID lists
2. **`build_task_mask(input_ids, task_token_ids)`** -- returns mask of 1s at positions matching the task instruction tokens
3. **Image mask construction** -- model-specific (see below)
4. **3-way pooling** in the main loop: `masked_mean_pool(hidden, mask)` applied with imgtext / image / task masks
5. **Save 3 files** at the end: `feats_A.pt`, `feats_A_img.pt`, `feats_A_txt.pt`

### Image mask strategy per architecture

| Architecture | Image mask method |
|---|---|
| StarVLA (Qwen2.5-VL / Qwen3-VL) | `input_ids == 151655` (IMAGE_TOKEN_INDEX) |
| SpatialVLA (PaLiGemma2) | `input_ids == 257152` |
| OpenVLA / CogACT (Prismatic) | Positional: positions `[1, 1+num_patches)` in multimodal sequence |
| Pi0 (PaliGemma Lerobot) | Positional: prefix positions `[0, n_img)` where prefix = [img_embs, lang_embs] |
| GR00T N1.6 (Eagle) | `backbone_outputs["image_mask"]` (already returned by backbone) |
| GR00T N1.5 (Eagle) | `eagle_input_ids == 151669` |

### Task instruction token strategy (all models)

Universal approach: `tokenizer.encode(" " + task_instruction, add_special_tokens=False)` then find as contiguous subsequence in the full `input_ids`.

The leading space (`" " + task`) is critical -- see Problem 1 below.

---

## Problems Encountered and Solutions

### Problem 1: BPE tokenization boundary (leading space)

**Symptom**: ~60% of samples triggered "WARNING: task tokens not found in input_ids". For non-empty instructions, the token IDs from `tokenizer.encode(task)` did not appear as a contiguous subsequence in the full `input_ids`.

**Root cause**: BPE tokenizers (Qwen, Llama, Gemma) encode the first character of a string differently depending on whether it starts a new segment or follows other text. Example with Qwen tokenizer:
- `encode("put the red object") -> [628, 279, 2518, 1633]` (token 628 = "put" without leading space)
- In the full chat template, "put" is preceded by " is ", so it becomes token 2182 = " put" (with space)

**Fix**: Use `tokenizer.encode(" " + instruction, add_special_tokens=False)` which produces the in-context token IDs. Verified empirically: `encode(" put the red object") -> [2182, 279, 2518, 1633]` matches the in-context tokens exactly.

**Status**: FIXED. Correct solution for all BPE tokenizers.

---

### Problem 2: Empty task descriptions (32% of N=5K)

**Symptom**: 1620 of 5000 samples have empty string `""` as task description.

**Root cause**: Some Bridge V2 episodes lack task annotations in the original dataset.

**Fix**: When task is empty, `task_ids = []` and `build_task_mask` returns an all-zero mask. The resulting `feats_A_txt` for those samples is a zero vector (0-dim mean pool falls back to zeros via `.clamp(min=1)` denominator).

**Status**: HANDLED, but **this is a concern for CKNNA quality**. 32% zero vectors in `feats_A_txt` will distort the kernel matrix. The CKNNA_txt results should be interpreted with this caveat -- or we should re-run excluding samples with empty descriptions.

---

### Problem 3: openvla-bridge-sft returns plain tensor

**Symptom**: `AttributeError: 'Tensor' object has no attribute 'hidden_states'` for the fine-tuned OpenVLA model.

**Root cause**: The `openvla-bridge-sft` checkpoint has a custom `PrismaticForConditionalGeneration.forward()` that:
1. Appends ACTION_DIM * NUM_ACTIONS_CHUNK placeholder tokens to input_ids
2. Calls `self.language_model(output_hidden_states=False)` -- hardcoded False
3. Returns `compute_logits` (a raw tensor), ignoring `output_hidden_states` and `return_dict` kwargs

The base `openvla-7b` uses the standard forward() and works fine.

**Fix**: Register hooks on `model.language_model`:
- Pre-hook: force `output_hidden_states=True, return_dict=True` in kwargs
- Post-hook: capture `output.hidden_states`

Also: the custom forward appends action tokens, making `hidden_states[-1]` longer than expected. Fix: compute `num_patches` from `model.vision_backbone.get_num_patches()` (deterministic) and trim `hidden_states` to `[1 + num_patches + text_seq_len]` before pooling.

**Status**: FIXED. Verified -- feats_A_img.pt and feats_A_txt.pt saved with shape (5000, 4096).

---

### Problem 4: CogACT-Small DiT model type

**Symptom**: `KeyError: 'DiT-Small'` when loading CogACT-Small.

**Root cause**: Orchestration script used `DiT-Small` but the CogACT action model registry uses `DiT-S`.

**Fix**: Changed `DiT-Small` -> `DiT-S` in `run_3way_5k.sh`.

**Status**: FIXED. Trivial typo.

---

### Problem 5: GR00T N1.5/N1.6 transformers version incompatibility

**Symptom**: `KeyError: 'resample'` in Eagle2.5 VL image processor's `preprocess()` method, and subsequently `TypeError: group_images_by_shape() missing 'disable_grouping'`.

**Root cause**: The `groot_libero` conda env had `transformers==4.51.3` (pinned by Isaac-GR00T), but someone upgraded it to 4.57.6 on 2026-02-27. The Eagle image processor was written for 4.51.3 API. In 4.57.6:
- `_prepare_input_images` was renamed to `_prepare_image_like_inputs`
- `_further_process_kwargs` now pops `resample` and converts to `interpolation` (Eagle's `preprocess` duplicated this logic)
- `group_images_by_shape` gained a required `disable_grouping` argument

Patch attempts (adding conditional kwargs handling in the Eagle processor) kept hitting cascading API changes.

**Fix**: Downgraded transformers back to 4.51.3 in the `groot_libero` env:
```
pip install transformers==4.51.3
```
This matches the version pinned in both `Isaac-GR00T/pyproject.toml` (N1.5) and `gr00t_1p6/Isaac-GR00T/pyproject.toml` (N1.6).

Also fixed GR00T N1.6 tokenizer loading: `AutoTokenizer.from_pretrained(args.ckpt)` failed because `Gr00tN1d6Config` isn't registered with AutoTokenizer. Changed to load from the Eagle model directory at `gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/`.

**Status**: FIXED. Both GR00T N1.5 and N1.6 extract successfully with shape (5000, 2048).

---

### Problem 6: Orchestration script syntax error at CKNNA section

**Symptom**: `run_3way_5k.sh: line 236: syntax error near unexpected token '('` after all extractions complete.

**Root cause**: `source conda/bin/activate <env>` in Sections 5-7 modified shell state (function definitions, shopt settings) in a way that broke bash's parsing of the `MODELS_3WAY=(...)` array syntax in Section 8.

**Fix**: Abandoned the single orchestration script for the CKNNA section. Ran models manually in separate tmux sessions per conda env. The CKNNA computation itself (Section 8) should be run as a standalone bash invocation.

**Status**: WORKAROUND. The extraction sections work fine; the CKNNA section needs to be run separately.

---

### Problem 7: Text mask fails for all non-StarVLA models (100% zero vectors)

**Symptom**: `feats_A_txt.pt` was 100% zero vectors for all 9 non-StarVLA models (spatialvla, pi0, openvla x2, cogact x3, groot x2). Only the 6 StarVLA models had non-zero text features (~39% zeros matching the 32% empty descriptions).

**Root cause**: `tokenizer.encode(" " + task, add_special_tokens=False)` produces a different first token than what appears in `input_ids` after the prompt template embeds the task. The leading space handling is tokenizer-dependent:

| Tokenizer | `encode(" put...")` | In-context (after template) | Match? |
|---|---|---|---|
| Gemma SentencePiece (SpatialVLA, Pi0) | `_put` (ID 2507) | `put` (ID 1065, after `<bos>`) | NO |
| Llama SentencePiece (OpenVLA, CogACT) | `_` + `_put` (two tokens) | `_put` (one token, after `"to "`) | NO |
| Eagle/Qwen3 BPE (GR00T) | `Gput` (ID 2182) | `put` (ID 628, after `\n`) | NO |
| Qwen2.5 BPE (StarVLA) | `Gput` (ID 2182) | `Gput` (ID 2182, after ` is `) | YES |

StarVLA worked because Qwen2.5's chat template places the task after a space (` is `), making the space-prefixed encoding correct. All other models place the task after `<bos>`, newline, or other non-space context.

**Fix**: Changed `build_task_mask` in all 7 scripts to try bare encoding first, then fall back to space-prefixed:
```python
for prefix in ["", " "]:
    task_ids = tokenizer.encode(prefix + task, add_special_tokens=False)
    start = find_subsequence(ids_list, task_ids)
    if start >= 0:
        mask[start:start + len(task_ids)] = 1
        return mask
```

**Status**: FIXED. All 9 non-StarVLA models re-extracted successfully.

---

### Problem 8: openvla-7b missing get_num_patches() method

**Symptom**: `AttributeError: 'PrismaticVisionBackbone' object has no attribute 'get_num_patches'` when running openvla-7b-bridge.

**Root cause**: `get_num_patches()` exists only in openvla-bridge-sft's custom code, not in base openvla-7b.

**Fix**: Added fallback: `vb.featurizer.patch_embed.num_patches` (= 256 for Prismatic fused backbone at 224px, patch 14).

**Status**: FIXED.

---

## Current Status (2026-02-27 12:01 UTC)

All extractions and CKNNA computations COMPLETE.

Text mask coverage: ~3377-3380/5000 (67.5%) for most models, groot-n16 = 2672 (53.4%). Zeros correspond to empty task descriptions in Bridge V2.

CKNNA results saved to `3way_cknna_results/` (9 JSON files covering 3 proprio variants, 3 action variants, and RT-1-X/Octo).

## Outputs Generated (2026-02-27 21:53 UTC)

CSV: `starVLA/cknna/record/cknna_3way_5k.csv` (17 models, 14 metric columns)

Plots in `starVLA/cknna/record/runs/3way_5k/`:
- scatter_CKNNA_proprio_3way.png (rho: imgtext=0.332, img=0.440, txt=-0.157)
- scatter_CKNNA_action_3way.png (all negative rho)
- scatter_MutualKNN_proprio_3way.png
- scatter_MutualKNN_action_3way.png
- bar_CKNNA_proprio_3way.png
- bar_CKNNA_action_3way.png
