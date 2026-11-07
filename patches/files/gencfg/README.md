# gencfg — recipe-owned generation config (2026-10-10)

`generation_config.json` = the checkpoint's own file (orcarouter/GLM-5.3-Flash-Uncensored-FP8@3cec42d6) with one change:
`<|assistant|>` (token 154828) appended to `eos_token_id` ([154820 <|endoftext|>, 154827 <|user|>, 154829 <|observation|>]
→ + 154828). vLLM turns every id in that list into a stop token (sampling_params.py update_from_generation_config, called
from v1/engine/input_processor.py), so a reply that starts a second assistant turn on its own ends there instead of
running on. Found by the MiaAI-Lab scan (their TensorFold patch 0075). Sampling defaults
(temperature 1.0, top_p 0.95) are unchanged. Used via `--generation-config /opt/recipe-patches/files/gencfg` (the patch
dir is synced to every node and mounted into every container).
