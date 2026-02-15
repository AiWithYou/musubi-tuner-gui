import glob
import gradio as gr
import inspect
import locale
import os
import re
import textwrap
import warnings
import toml
import subprocess
import sys
import json
import shlex
from config_manager import ConfigManager
from i18n_data import I18N_DATA

config_manager = ConfigManager()


def _create_fallback_i18n():
    default_lang = "en"
    try:
        lang_code = (locale.getdefaultlocale() or [""])[0] or ""
        if lang_code.lower().startswith("ja"):
            default_lang = "ja"
    except Exception:
        default_lang = "en"

    def _lookup(key):
        if default_lang in I18N_DATA and key in I18N_DATA[default_lang]:
            return I18N_DATA[default_lang][key]
        return I18N_DATA["en"].get(key, key)

    return _lookup


if hasattr(gr, "I18n"):
    i18n = gr.I18n(en=I18N_DATA["en"], ja=I18N_DATA["ja"])
else:
    i18n = _create_fallback_i18n()

try:
    _launch_params = inspect.signature(gr.Blocks.launch).parameters
except (ValueError, TypeError):
    _launch_params = {}

LAUNCH_SUPPORTS_THEME = "theme" in _launch_params
LAUNCH_SUPPORTS_CSS = "css" in _launch_params
LAUNCH_SUPPORTS_I18N = "i18n" in _launch_params

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="The 'theme' parameter in the Blocks constructor will be removed in Gradio 6.0.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="The 'css' parameter in the Blocks constructor will be removed in Gradio 6.0.*",
)

APP_CSS = """
@import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap");

:root {
  --bg-1: #f7f1e3;
  --bg-2: #e9f0ff;
  --bg-3: #fcefdc;
  --card: #fffdf8;
  --ink: #1f1b16;
  --muted: #5b5a57;
  --accent: #0f766e;
  --accent-project: #2563eb;
  --accent-preset: #f97316;
  --accent-path: #16a34a;
  --accent-model: #0284c7;
  --accent-training: #ef4444;
  --border: rgba(31, 41, 55, 0.16);
  --border-strong: rgba(31, 41, 55, 0.28);
  --shadow: 0 20px 50px rgba(15, 23, 42, 0.12);
  --shadow-soft: 0 10px 25px rgba(15, 23, 42, 0.08);
}

.gradio-container {
  font-family: "Space Grotesk", "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Meiryo", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(1200px 600px at 90% -20%, rgba(37, 99, 235, 0.15), transparent 70%),
    radial-gradient(900px 500px at -10% 10%, rgba(249, 115, 22, 0.18), transparent 70%),
    radial-gradient(800px 400px at 50% 120%, rgba(16, 185, 129, 0.16), transparent 60%),
    linear-gradient(180deg, #faf7f0 0%, #f1f5f9 100%);
  min-height: 100vh;
}

.gradio-container:before,
.gradio-container:after {
  content: "";
  position: fixed;
  z-index: 0;
  pointer-events: none;
  border-radius: 999px;
  filter: blur(20px);
  opacity: 0.5;
}

.gradio-container:before {
  width: 420px;
  height: 420px;
  top: -140px;
  right: 8%;
  background: rgba(37, 99, 235, 0.22);
}

.gradio-container:after {
  width: 520px;
  height: 520px;
  bottom: -220px;
  left: -120px;
  background: rgba(249, 115, 22, 0.2);
}

#app-header h1 {
  font-weight: 700;
  letter-spacing: -0.03em;
  margin-bottom: 6px;
}

#app-header,
#app-desc {
  position: relative;
  z-index: 1;
  animation: fade-in 0.6s ease-out both;
}

#app-desc {
  color: var(--muted);
  margin-bottom: 16px;
  max-width: 900px;
}

#main-tabs {
  position: relative;
  z-index: 1;
}

#main-tabs .tab-nav {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  padding: 6px;
  margin-bottom: 12px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.6);
  box-shadow: var(--shadow-soft);
}

#main-tabs .tab-nav button {
  border: 1px solid transparent;
  padding: 6px 14px;
  border-radius: 999px;
  font-weight: 600;
  background: transparent;
  color: var(--muted);
  transition: all 0.2s ease;
}

#main-tabs .tab-nav button:hover {
  color: var(--ink);
  border-color: var(--border-strong);
  background: rgba(255, 255, 255, 0.9);
}

#main-tabs .tab-nav button.selected,
#main-tabs .tab-nav button.active,
#main-tabs .tab-nav button[aria-selected="true"] {
  color: var(--ink);
  border-color: var(--border-strong);
  background: #ffffff;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.12);
}

.section-card {
  background: var(--card);
  border: 1.5px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: var(--shadow);
  animation: fade-in 0.4s ease-out both;
}

.section-card h2,
.section-card h3,
.section-card h4 {
  margin: 0 0 8px 0;
}

.section-card + .section-card {
  margin-top: 12px;
}

.section-card.card-preset {
  border-left: 6px solid var(--accent-preset);
}

.section-card.card-project {
  border-left: 6px solid var(--accent-project);
}

.section-card.card-model {
  border-left: 6px solid var(--accent-model);
}

.section-card.card-preprocess {
  border-left: 6px solid var(--accent-path);
}

.section-card.card-training {
  border-left: 6px solid var(--accent-training);
}

.section-card.card-post {
  border-left: 6px solid #64748b;
}

.gr-button {
  border-radius: 12px;
  border: 1px solid var(--border-strong);
  font-weight: 600;
  transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}

.gr-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
}

.gr-button.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}

.gr-button.primary:hover {
  background: #0d6e66;
}

.gr-text-input input,
.gr-text-input textarea,
.gr-textbox textarea,
.gr-number input {
  border-radius: 10px;
  border: 1px solid var(--border);
}

.gr-group,
.gr-row,
.gr-column,
.section-card {
  overflow: visible;
  position: relative;
}

.gr-dropdown {
  position: relative;
  z-index: 5;
}

.gr-dropdown .options,
.gr-dropdown .choices,
.gr-dropdown .dropdown {
  z-index: 50;
}

.path-row {
  align-items: flex-end;
  gap: 12px;
}

.project-row {
  border-left: 4px solid var(--accent-project);
  border-radius: 12px;
  padding-left: 10px;
  background: linear-gradient(90deg, rgba(37, 99, 235, 0.12), rgba(37, 99, 235, 0));
}

.env-row {
  border-left: 4px solid var(--accent-path);
  border-radius: 12px;
  padding-left: 10px;
  background: linear-gradient(90deg, rgba(22, 163, 74, 0.12), rgba(22, 163, 74, 0));
}

.context-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-bottom: 10px;
}

.context-legend .tag {
  display: inline-flex;
  align-items: center;
  padding: 4px 12px;
  border-radius: 999px;
  border: 1px solid var(--border);
  font-size: 0.8rem;
  font-weight: 600;
}

.context-legend > * {
  flex: 0 0 auto;
}

.context-legend .gr-markdown p {
  margin: 0;
}

.context-legend .tag.preset {
  background: #fff1e7;
  border-color: #fdba74;
  color: #9a3412;
}

.context-legend .tag.project {
  background: #eaf0ff;
  border-color: #93c5fd;
  color: #1d4ed8;
}

.context-legend .tag.path {
  background: #e9fbe9;
  border-color: #86efac;
  color: #166534;
}

.context-legend .legend-note {
  color: var(--muted);
  font-size: 0.9rem;
}

.subtle-note {
  color: var(--muted);
  font-size: 0.9rem;
}

@keyframes fade-in {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

@media (max-width: 900px) {
  #main-tabs .tab-nav {
    border-radius: 16px;
  }
  .path-row {
    flex-direction: column;
    align-items: stretch;
  }
  .gr-button {
    width: 100%;
  }
}
"""

def construct_ui():
    # --- Preset Management ---
    PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")
    os.makedirs(PRESETS_DIR, exist_ok=True)
    TRAINING_MODE_LORA = "LoRA/LoHa/LoKr"
    TRAINING_MODE_FINETUNE = "Fine-tune"

    def _normalize_model_label(model_name):
        return "Z-Image" if model_name == "Z-Image-Turbo" else model_name

    def _is_zimage(model_name):
        return _normalize_model_label(model_name) == "Z-Image"

    def ensure_default_presets():
        zimage_vram = "24"
        zimage_resolution = config_manager.get_resolution("Z-Image")
        zimage_batch = config_manager.get_batch_size("Z-Image", zimage_vram)
        zimage_defaults = config_manager.get_training_defaults("Z-Image", zimage_vram, "")

        preset = {
            "project_dir": "",
            "model_arch": "Z-Image",
            "vram_size": zimage_vram,
            "comfy_models_dir": "",
            "resolution_w": zimage_resolution[0],
            "resolution_h": zimage_resolution[1],
            "batch_size": zimage_batch,
            "control_directory": "",
            "control_resolution_w": 0,
            "control_resolution_h": 0,
            "no_resize_control": False,
            "image_directory": "",
            "cache_directory": "",
            "caption_extension": ".txt",
            "num_repeats": 1,
            "enable_bucket": True,
            "bucket_no_upscale": False,
            "vae_path": "",
            "text_encoder1_path": "",
            "text_encoder2_path": "",
            "dit_path": zimage_defaults.get("dit_path", ""),
            "output_name": "zimage_lora",
            "dim": zimage_defaults.get("network_dim", 4),
            "lr": zimage_defaults.get("learning_rate", 1e-3),
            "optimizer_type": zimage_defaults.get("optimizer_type", "adamw8bit"),
            "optimizer_args": zimage_defaults.get("optimizer_args", ""),
            "lr_scheduler": "constant",
            "lr_scheduler_args": "",
            "network_alpha": zimage_defaults.get("network_alpha", 1),
            "lr_warmup_steps": zimage_defaults.get("lr_warmup_steps", 0),
            "seed": zimage_defaults.get("seed", 42),
            "max_grad_norm": zimage_defaults.get("max_grad_norm", 1.0),
            "epochs": 16,
            "save_every": zimage_defaults.get("save_every_n_epochs", 1),
            "flow_shift": zimage_defaults.get("discrete_flow_shift", 2.0),
            "block_swap": zimage_defaults.get("block_swap", 0),
            "pinned": zimage_defaults.get("use_pinned_memory_for_block_swap", False),
            "mixed_precision": zimage_defaults.get("mixed_precision", "bf16"),
            "grad_checkpointing": zimage_defaults.get("gradient_checkpointing", True),
            "fp8_scaled": zimage_defaults.get("fp8_scaled", True),
            "fp8_llm": zimage_defaults.get("fp8_llm", False),
            "additional_args": "",
            "sample_images": False,
            "sample_every": 1,
            "sample_output_dir": "",
            "sample_prompt": "",
            "sample_neg": "",
            "sample_w": zimage_resolution[0],
            "sample_h": zimage_resolution[1],
            "input_lora": "",
            "output_comfy": "",
            "training_mode": zimage_defaults.get("training_mode", TRAINING_MODE_LORA),
            "network_type": zimage_defaults.get("network_type", "LoRA"),
            "network_args": zimage_defaults.get("network_args", ""),
            "full_bf16": zimage_defaults.get("full_bf16", False),
            "fused_backward_pass": zimage_defaults.get("fused_backward_pass", False),
            "mem_eff_save": zimage_defaults.get("mem_eff_save", False),
            "block_swap_optimizer_patch_params": zimage_defaults.get("block_swap_optimizer_patch_params", False),
            "lokr_rank": "",
        }

        defaults = {"Z-Image (default)": preset}
        for name, data in defaults.items():
            path = os.path.join(PRESETS_DIR, f"{name}.json")
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

    ensure_default_presets()
    initial_fp8_llm = config_manager.get_training_defaults("Flux.2 Klein (4B)", "24", "").get("fp8_llm", False)

    components = {}

    def register(name, component):
        components[name] = component
        return component

    def components_for(keys):
        return [components[key] for key in keys]

    def pack_updates(keys, updates, default_factory=gr.update):
        packed = []
        for key in keys:
            if key in updates:
                packed.append(updates[key])
            else:
                packed.append(default_factory())
        return packed

    def tab_label(text):
        if not text:
            return ""
        label = re.sub(r"^#+\s*", "", str(text)).strip()
        label = re.sub(r"^\d+\s*[\.\)\-:]\s*", "", label).strip()
        return label

    PRESET_FIELD_MAP = {
        "project_dir": "project_dir",
        "model_arch": "model_arch",
        "vram_size": "vram_size",
        "comfy_models_dir": "comfy_models_dir",
        "resolution_w": "resolution_w",
        "resolution_h": "resolution_h",
        "batch_size": "batch_size",
        "control_directory": "control_directory",
        "control_resolution_w": "control_res_w",
        "control_resolution_h": "control_res_h",
        "no_resize_control": "no_resize_control",
        "image_directory": "image_directory",
        "cache_directory": "cache_directory",
        "caption_extension": "caption_extension",
        "num_repeats": "num_repeats",
        "enable_bucket": "enable_bucket",
        "bucket_no_upscale": "bucket_no_upscale",
        "vae_path": "vae_path",
        "text_encoder1_path": "text_encoder1_path",
        "text_encoder2_path": "text_encoder2_path",
        "dit_path": "dit_path",
        "output_name": "output_name",
        "dim": "network_dim",
        "lr": "learning_rate",
        "optimizer_type": "optimizer_type",
        "optimizer_args": "optimizer_args",
        "lr_scheduler": "lr_scheduler",
        "lr_scheduler_args": "lr_scheduler_args",
        "network_alpha": "network_alpha",
        "lr_warmup_steps": "lr_warmup_steps",
        "seed": "seed",
        "max_grad_norm": "max_grad_norm",
        "epochs": "num_epochs",
        "save_every": "save_every_n_epochs",
        "flow_shift": "discrete_flow_shift",
        "block_swap": "block_swap",
        "pinned": "use_pinned_memory_for_block_swap",
        "mixed_precision": "mixed_precision",
        "grad_checkpointing": "gradient_checkpointing",
        "fp8_scaled": "fp8_scaled",
        "fp8_llm": "fp8_llm",
        "additional_args": "additional_args",
        "sample_images": "sample_images",
        "sample_every": "sample_every_n",
        "sample_output_dir": "sample_output_dir",
        "sample_prompt": "sample_prompt",
        "sample_neg": "sample_negative_prompt",
        "sample_w": "sample_w",
        "sample_h": "sample_h",
        "input_lora": "input_lora",
        "output_comfy": "output_comfy_lora",
        "training_mode": "training_mode",
        "network_type": "network_type",
        "network_args": "network_args",
        "full_bf16": "full_bf16",
        "fused_backward_pass": "fused_backward_pass",
        "mem_eff_save": "mem_eff_save",
        "block_swap_optimizer_patch_params": "block_swap_optimizer_patch_params",
        "lokr_rank": "lokr_rank",
    }

    PRESET_DATA_KEYS = [
        "project_dir",
        "model_arch",
        "vram_size",
        "comfy_models_dir",
        "resolution_w",
        "resolution_h",
        "batch_size",
        "control_directory",
        "control_resolution_w",
        "control_resolution_h",
        "no_resize_control",
        "image_directory",
        "cache_directory",
        "caption_extension",
        "num_repeats",
        "enable_bucket",
        "bucket_no_upscale",
        "vae_path",
        "text_encoder1_path",
        "text_encoder2_path",
        "dit_path",
        "output_name",
        "dim",
        "lr",
        "optimizer_type",
        "optimizer_args",
        "lr_scheduler",
        "lr_scheduler_args",
        "network_alpha",
        "lr_warmup_steps",
        "seed",
        "max_grad_norm",
        "epochs",
        "save_every",
        "flow_shift",
        "block_swap",
        "pinned",
        "mixed_precision",
        "grad_checkpointing",
        "fp8_scaled",
        "fp8_llm",
        "additional_args",
        "sample_images",
        "sample_every",
        "sample_output_dir",
        "sample_prompt",
        "sample_neg",
        "sample_w",
        "sample_h",
        "input_lora",
        "output_comfy",
        "training_mode",
        "network_type",
        "network_args",
        "full_bf16",
        "fused_backward_pass",
        "mem_eff_save",
        "block_swap_optimizer_patch_params",
        "lokr_rank",
    ]
    PRESET_COMPONENT_KEYS = [PRESET_FIELD_MAP[key] for key in PRESET_DATA_KEYS]
    PRESET_OUTPUT_COMPONENT_KEYS = PRESET_COMPONENT_KEYS

    INIT_OUTPUT_KEYS = [
        "model_arch",
        "vram_size",
        "comfy_models_dir",
        "resolution_w",
        "resolution_h",
        "batch_size",
        "control_directory",
        "control_res_w",
        "control_res_h",
        "no_resize_control",
        "image_directory",
        "cache_directory",
        "caption_extension",
        "num_repeats",
        "enable_bucket",
        "bucket_no_upscale",
        "toml_preview",
        "vae_path",
        "text_encoder1_path",
        "text_encoder2_path",
        "dit_path",
        "output_name",
        "network_dim",
        "learning_rate",
        "optimizer_type",
        "optimizer_args",
        "lr_scheduler",
        "lr_scheduler_args",
        "network_alpha",
        "lr_warmup_steps",
        "seed",
        "max_grad_norm",
        "num_epochs",
        "save_every_n_epochs",
        "discrete_flow_shift",
        "block_swap",
        "use_pinned_memory_for_block_swap",
        "mixed_precision",
        "gradient_checkpointing",
        "fp8_scaled",
        "fp8_llm",
        "additional_args",
        "sample_images",
        "sample_every_n",
        "sample_output_dir",
        "sample_prompt",
        "sample_negative_prompt",
        "sample_w",
        "sample_h",
        "input_lora",
        "output_comfy_lora",
        "training_mode",
        "network_type",
        "network_args",
        "full_bf16",
        "fused_backward_pass",
        "mem_eff_save",
        "block_swap_optimizer_patch_params",
        "lokr_rank",
    ]

    TRAINING_DEFAULT_KEYS = [
        "dit_path",
        "network_dim",
        "learning_rate",
        "optimizer_type",
        "optimizer_args",
        "training_mode",
        "network_type",
        "network_args",
        "network_alpha",
        "lr_warmup_steps",
        "seed",
        "max_grad_norm",
        "num_epochs",
        "save_every_n_epochs",
        "discrete_flow_shift",
        "block_swap",
        "use_pinned_memory_for_block_swap",
        "mixed_precision",
        "gradient_checkpointing",
        "fp8_scaled",
        "fp8_llm",
        "full_bf16",
        "fused_backward_pass",
        "mem_eff_save",
        "block_swap_optimizer_patch_params",
        "sample_every_n",
        "sample_w",
        "sample_h",
        "sample_output_dir",
    ]

    QUICK_SETUP_OUTPUT_KEYS = [
        "resolution_w",
        "resolution_h",
        "batch_size",
        "vae_path",
        "text_encoder1_path",
        "text_encoder2_path",
        "dit_path",
        "network_dim",
        "learning_rate",
        "optimizer_type",
        "optimizer_args",
        "training_mode",
        "network_type",
        "network_args",
        "network_alpha",
        "lr_warmup_steps",
        "seed",
        "max_grad_norm",
        "num_epochs",
        "save_every_n_epochs",
        "discrete_flow_shift",
        "block_swap",
        "use_pinned_memory_for_block_swap",
        "mixed_precision",
        "gradient_checkpointing",
        "fp8_scaled",
        "fp8_llm",
        "full_bf16",
        "fused_backward_pass",
        "mem_eff_save",
        "block_swap_optimizer_patch_params",
        "sample_every_n",
        "sample_w",
        "sample_h",
        "sample_output_dir",
        "training_model_info",
        "quick_status",
    ]

    RECOMMENDED_KEYS = ["resolution_w", "resolution_h", "batch_size"]

    def get_preset_list():
        return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(PRESETS_DIR, "*.json"))]

    def save_preset(name, *args):
        if not name:
            return i18n("msg_preset_error").format(e="Name is empty")
        try:
            # args order corresponds to the inputs list of save button
            data = dict(zip(PRESET_DATA_KEYS, args))
            path = os.path.join(PRESETS_DIR, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return i18n("msg_preset_saved").format(name=name)
        except Exception as e:
            return i18n("msg_preset_error").format(e=str(e))

    def load_preset(name, apply_paths):
        empty_updates = pack_updates(PRESET_OUTPUT_COMPONENT_KEYS, {})
        if not name:
            return [i18n("msg_preset_error").format(e="Name is empty"), *empty_updates]
        try:
            path = os.path.join(PRESETS_DIR, f"{name}.json")
            if not os.path.exists(path):
                return [i18n("msg_preset_error").format(e="Preset not found"), *empty_updates]
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            def _looks_shifted_preset(payload):
                if "sample_output_dir" not in payload:
                    return False
                if "output_comfy" in payload:
                    return False
                sample_h_val = payload.get("sample_h")
                sample_neg_val = payload.get("sample_neg")
                if isinstance(sample_h_val, (int, float)):
                    return False
                if isinstance(sample_neg_val, (int, float)):
                    return True
                return False

            def _repair_shifted_preset(payload):
                fixed = dict(payload)
                fixed["sample_prompt"] = payload.get("sample_output_dir", "")
                fixed["sample_neg"] = payload.get("sample_prompt", "")
                fixed["sample_w"] = payload.get("sample_neg", 1024)
                fixed["sample_h"] = payload.get("sample_w", 1024)
                fixed["input_lora"] = payload.get("sample_h", "")
                fixed["output_comfy"] = payload.get("input_lora", "")
                fixed["sample_output_dir"] = ""
                return fixed

            repaired = False
            if _looks_shifted_preset(data):
                data = _repair_shifted_preset(data)
                repaired = True

            path_keys = {
                "comfy_models_dir",
                "control_directory",
                "image_directory",
                "cache_directory",
                "vae_path",
                "text_encoder1_path",
                "text_encoder2_path",
                "dit_path",
                "sample_output_dir",
                "input_lora",
                "output_comfy",
            }

            status = i18n("msg_preset_loaded").format(name=name)
            if not apply_paths:
                status = f"{status}\n\n{i18n('msg_preset_paths_kept')}"
            if repaired:
                status = f"{status}\n\n{i18n('msg_preset_repaired')}"

            updates = {}
            if "model_arch" in data:
                updates["model_arch"] = _normalize_model_label(data.get("model_arch", "Flux.2 Klein (4B)"))

            for data_key, component_key in PRESET_FIELD_MAP.items():
                if data_key in {"project_dir", "model_arch"}:
                    continue
                if data_key not in data:
                    continue
                if (not apply_paths) and data_key in path_keys:
                    continue
                updates[component_key] = data.get(data_key)

            # Backward compatibility: older presets may miss newer training/conversion keys.
            missing_key_defaults = {
                "training_mode": TRAINING_MODE_LORA,
                "network_type": "LoRA",
                "network_args": "",
                "full_bf16": False,
                "fused_backward_pass": False,
                "mem_eff_save": False,
                "block_swap_optimizer_patch_params": False,
                "lokr_rank": "",
            }
            for data_key, default_value in missing_key_defaults.items():
                component_key = PRESET_FIELD_MAP[data_key]
                if data_key not in data and component_key not in updates:
                    updates[component_key] = default_value

            effective_model = updates.get("model_arch", _normalize_model_label(data.get("model_arch", "Flux.2 Klein (4B)")))
            if not _is_zimage(effective_model):
                updates["training_mode"] = TRAINING_MODE_LORA

            return [status, *pack_updates(PRESET_OUTPUT_COMPONENT_KEYS, updates)]
        except Exception as e:
            # In case of error, just don't update anything or handle gracefully
            print(f"Error loading preset: {e}")
            return [i18n("msg_preset_error").format(e=str(e)), *empty_updates]

    def refresh_preset_dropdown():
        return gr.update(choices=get_preset_list())

    def _safe_initial_dir(path):
        if not path:
            return None
        if os.path.isdir(path):
            return path
        if os.path.isfile(path):
            return os.path.dirname(path)
        parent = os.path.dirname(path)
        return parent if parent and os.path.exists(parent) else None

    def browse_file(current_path):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            initialdir = _safe_initial_dir(current_path)
            filename = filedialog.askopenfilename(initialdir=initialdir)
            root.destroy()
            return filename if filename else gr.update()
        except Exception as e:
            return gr.update()

    def browse_dir(current_path):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            initialdir = _safe_initial_dir(current_path)
            dirname = filedialog.askdirectory(initialdir=initialdir)
            root.destroy()
            return dirname if dirname else gr.update()
        except Exception:
            return gr.update()

    def browse_save(current_path):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            initialdir = _safe_initial_dir(current_path)
            filename = filedialog.asksaveasfilename(initialdir=initialdir)
            root.destroy()
            return filename if filename else gr.update()
        except Exception:
            return gr.update()

    def open_folder(path):
        target = (path or "").strip()
        if not target:
            return i18n("msg_open_path_missing").format(path=path or "")
        if not os.path.exists(target):
            return i18n("msg_open_path_missing").format(path=target)
        try:
            if os.name == "nt":
                os.startfile(target)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
            return i18n("msg_open_path_ok").format(path=target)
        except Exception as e:
            return i18n("msg_open_path_fail").format(path=target, e=str(e))

    def open_project_folder(path):
        return open_folder(path)

    def open_training_folder(path):
        return open_folder(os.path.join(path, "training") if path else "")

    def open_logs_folder(path):
        return open_folder(os.path.join(path, "logs") if path else "")

    def open_presets_folder():
        return open_folder(PRESETS_DIR)

    def launch_tensorboard(project_dir):
        if not project_dir or not os.path.exists(project_dir):
            return "Error: Project directory invalid."
        log_dir = os.path.join(project_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        try:
            # Launch in background
            subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", "6006"], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return i18n("msg_tensorboard")
        except Exception as e:
            return f"Error launching TensorBoard: {e}"

    # --- UI Construction ---
    # I18N doesn't work for gr.Blocks title
    # with gr.Blocks(title=i18n("app_title")) as demo:
    block_kwargs = {"title": "Musubi Tuner GUI"}
    if not LAUNCH_SUPPORTS_THEME:
        block_kwargs["theme"] = gr.themes.Soft()
    if not LAUNCH_SUPPORTS_CSS:
        block_kwargs["css"] = APP_CSS

    with gr.Blocks(**block_kwargs) as demo:
        gr.Markdown(i18n("app_header"), elem_id="app-header")
        gr.Markdown(i18n("app_desc"), elem_id="app-desc")
        with gr.Tabs(elem_id="main-tabs"):

            with gr.TabItem(tab_label(i18n("acc_project"))):
                # Presets Section
                with gr.Group(elem_classes=["section-card", "card-preset"]):
                    gr.Markdown(i18n("header_presets"))
                    gr.Markdown(i18n("desc_preset_scope"), elem_classes=["subtle-note"])
                    with gr.Row(elem_classes=["context-legend"]):
                        gr.Markdown(i18n("tag_preset"), elem_classes=["tag", "preset"])
                        gr.Markdown(i18n("tag_project"), elem_classes=["tag", "project"])
                        gr.Markdown(i18n("tag_paths"), elem_classes=["tag", "path"])
                    gr.Markdown(i18n("header_preset_save"))
                    with gr.Row():
                        preset_name = gr.Textbox(label=i18n("lbl_preset_name"), scale=2)
                        save_preset_btn = gr.Button(i18n("btn_save_preset"), scale=1)
                    gr.Markdown(i18n("header_preset_load"))
                    with gr.Row():
                        load_preset_dd = gr.Dropdown(label=i18n("lbl_load_preset"), choices=get_preset_list(), scale=2)
                        load_preset_btn = gr.Button(i18n("btn_load_preset"), scale=1)
                        refresh_preset_btn = gr.Button(i18n("btn_refresh_presets"), scale=0)
                    with gr.Row():
                        preset_apply_paths = gr.Checkbox(label=i18n("lbl_preset_apply_paths"), value=False, scale=2)
                        open_presets_btn = gr.Button(i18n("btn_open_presets"), scale=1)
                    preset_status = gr.Markdown("")
        
                with gr.Group(elem_classes=["section-card", "card-project"]):
                    gr.Markdown(i18n("acc_project"))
                    gr.Markdown(i18n("desc_project"))
                    with gr.Row(elem_classes=["path-row", "project-row"]):
                        project_dir = gr.Textbox(label=i18n("lbl_proj_dir"), placeholder=i18n("ph_proj_dir"), max_lines=1, scale=6)
                        browse_project_dir = gr.Button(i18n("btn_browse"), scale=1)
                        open_project_btn = gr.Button(i18n("btn_open_project"), scale=1)
                    with gr.Row():
                        open_training_btn = gr.Button(i18n("btn_open_training"))
                        open_logs_btn = gr.Button(i18n("btn_open_logs"))
        
                    # Placeholder for project initialization or loading
                    init_btn = gr.Button(i18n("btn_init_project"))
                    project_status = gr.Markdown("")
        
            with gr.TabItem(tab_label(i18n("acc_model"))):
                with gr.Group(elem_classes=["section-card", "card-model"]):
                    gr.Markdown(i18n("acc_model"))
                    gr.Markdown(i18n("desc_model"))
                    with gr.Row():
                        model_arch = gr.Dropdown(
                            label=i18n("lbl_model_arch"),
                            choices=[
                                "Flux.2 Klein (4B)",
                                "Flux.2 Klein Base (4B)",
                                "Flux.2 Dev",
                                "Qwen-Image",
                                "Z-Image",
                            ],
                            value="Flux.2 Klein (4B)",
                        )
                        vram_size = gr.Dropdown(label=i18n("lbl_vram"), choices=["12", "16", "24", "32", ">32"], value="24")
        
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        comfy_models_dir = gr.Textbox(label=i18n("lbl_comfy_dir"), placeholder=i18n("ph_comfy_dir"), max_lines=1, scale=8)
                        browse_comfy_dir = gr.Button(i18n("btn_browse"), scale=1)
        
                    # Validation for ComfyUI models directory
                    models_status = gr.Markdown("")
                    validate_models_btn = gr.Button(i18n("btn_validate_models"))
        
                    gr.Markdown(i18n("header_quick_actions"))
                    with gr.Row():
                        quick_setup_btn = gr.Button(i18n("btn_quick_setup"), variant="primary")
                        check_missing_btn = gr.Button(i18n("btn_check_missing"))
                    quick_status = gr.Markdown("", elem_classes=["subtle-note"])
        
        
                    # Placeholder for Dataset Settings (Step 3)
                    gr.Markdown(i18n("header_dataset"))
                    gr.Markdown(i18n("desc_dataset"))
                    with gr.Row():
                        set_rec_settings_btn = gr.Button(i18n("btn_rec_res_batch"))
                    with gr.Row():
                        resolution_w = gr.Number(label=i18n("lbl_res_w"), value=1024, precision=0)
                        resolution_h = gr.Number(label=i18n("lbl_res_h"), value=1024, precision=0)
                        batch_size = gr.Number(label=i18n("lbl_batch_size"), value=1, precision=0)
        
                    gr.Markdown(i18n("header_control"))
                    gr.Markdown(i18n("desc_control"))
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        control_directory = gr.Textbox(label=i18n("lbl_control_dir"), placeholder=i18n("ph_control_dir"), max_lines=1, scale=8)
                        browse_control_dir = gr.Button(i18n("btn_browse"), scale=1)
                    with gr.Row():
                        control_res_w = gr.Number(label=i18n("lbl_control_res_w"), value=0, precision=0)
                        control_res_h = gr.Number(label=i18n("lbl_control_res_h"), value=0, precision=0)
                        no_resize_control = gr.Checkbox(label=i18n("lbl_no_resize_control"), value=False)
        
                    gr.Markdown(i18n("header_dataset_details"))
                    gr.Markdown(i18n("desc_dataset_details"))
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        image_directory = gr.Textbox(label=i18n("lbl_image_dir"), placeholder=i18n("ph_image_dir"), max_lines=1, scale=8)
                        browse_image_dir = gr.Button(i18n("btn_browse"), scale=1)
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        cache_directory = gr.Textbox(label=i18n("lbl_cache_dir"), placeholder=i18n("ph_cache_dir"), max_lines=1, scale=8)
                        browse_cache_dir = gr.Button(i18n("btn_browse"), scale=1)
                    with gr.Row():
                        caption_extension = gr.Textbox(label=i18n("lbl_caption_ext"), value=".txt", max_lines=1)
                        num_repeats = gr.Number(label=i18n("lbl_num_repeats"), value=1, precision=0)
                    with gr.Row():
                        enable_bucket = gr.Checkbox(label=i18n("lbl_enable_bucket"), value=True)
                        bucket_no_upscale = gr.Checkbox(label=i18n("lbl_bucket_no_upscale"), value=False)
        
                    gen_toml_btn = gr.Button(i18n("btn_gen_config"))
                    dataset_status = gr.Markdown("")
                    toml_preview = gr.Code(label=i18n("lbl_toml_preview"), interactive=False)
        
                    def load_project_settings(project_path):
                        settings = {}
                        try:
                            settings_path = os.path.join(project_path, "musubi_project.toml")
                            if os.path.exists(settings_path):
                                with open(settings_path, "r", encoding="utf-8") as f:
                                    settings = toml.load(f)
                        except Exception as e:
                            print(f"Error loading project settings: {e}")
                        return settings
        
                    def load_dataset_config_content(project_path):
                        content = ""
                        try:
                            config_path = os.path.join(project_path, "dataset_config.toml")
                            if os.path.exists(config_path):
                                with open(config_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                        except Exception as e:
                            print(f"Error reading dataset config: {e}")
                        return content
        
                    def save_project_settings(project_path, **kwargs):
                        try:
                            # Load existing settings to support partial updates
                            settings = load_project_settings(project_path)
                            # Update with new values
                            settings.update(kwargs)
        
                            settings_path = os.path.join(project_path, "musubi_project.toml")
                            with open(settings_path, "w", encoding="utf-8") as f:
                                toml.dump(settings, f)
                        except Exception as e:
                            print(f"Error saving project settings: {e}")
        
                    def init_project(path):
                        if not path:
                            return ("Please enter a project directory path.", *pack_updates(INIT_OUTPUT_KEYS, {}))
                        try:
                            os.makedirs(os.path.join(path, "training"), exist_ok=True)
        
                            # Load settings if available
                            settings = load_project_settings(path)
                            if not settings:
                                preview_content = load_dataset_config_content(path)
                                msg = f"Project initialized at {path}. No saved settings found; keeping current values."
                                msg += "\n\nProject initialized. No saved settings were found, so current form values were kept."
                                updates = {}
                                if preview_content:
                                    updates["toml_preview"] = preview_content
                                return (msg, *pack_updates(INIT_OUTPUT_KEYS, updates))
                            new_model = _normalize_model_label(settings.get("model_arch", "Flux.2 Klein (4B)"))
                            new_vram = settings.get("vram_size", "16")
                            new_comfy = settings.get("comfy_models_dir", "")
                            new_w = settings.get("resolution_w", 1024)
                            new_h = settings.get("resolution_h", 1024)
                            new_batch = settings.get("batch_size", 1)
                            new_control_dir = settings.get("control_directory", "")
                            new_control_w = settings.get("control_resolution_w", 0)
                            new_control_h = settings.get("control_resolution_h", 0)
                            new_no_resize_control = settings.get("no_resize_control", False)
                            new_image_dir = settings.get("image_directory") or os.path.join(path, "training")
                            new_cache_dir = settings.get("cache_directory") or os.path.join(path, "cache")
                            new_caption_ext = settings.get("caption_extension", ".txt")
                            new_num_repeats = settings.get("num_repeats", 1)
                            new_enable_bucket = settings.get("enable_bucket", True)
                            new_bucket_no_upscale = settings.get("bucket_no_upscale", False)
                            new_vae = settings.get("vae_path", "")
                            new_te1 = settings.get("text_encoder1_path", "")
                            new_te2 = settings.get("text_encoder2_path", "")
        
                            # Training params
                            new_dit = settings.get("dit_path", "")
                            new_out_nm = settings.get("output_name", "my_lora")
                            new_dim = settings.get("network_dim", 4)
                            new_lr = settings.get("learning_rate", 1e-4)
                            new_optimizer_type = settings.get("optimizer_type", "adamw8bit")
                            new_optimizer_args = settings.get("optimizer_args", "")
                            new_network_alpha = settings.get("network_alpha", 1)
                            new_lr_warmup_steps = settings.get("lr_warmup_steps", 0)
                            new_seed = settings.get("seed", 42)
                            new_max_grad_norm = settings.get("max_grad_norm", 1.0)
                            new_lr_scheduler = settings.get("lr_scheduler", "constant")
                            new_lr_scheduler_args = settings.get("lr_scheduler_args", "")
                            new_training_mode = settings.get("training_mode", TRAINING_MODE_LORA)
                            if not _is_zimage(new_model):
                                new_training_mode = TRAINING_MODE_LORA
                            new_network_type = settings.get("network_type", "LoRA")
                            new_network_args = settings.get("network_args", "")
                            new_epochs = settings.get("num_epochs", 16)
                            new_save_n = settings.get("save_every_n_epochs", 1)
                            new_flow = settings.get("discrete_flow_shift", 2.0)
                            new_swap = settings.get("block_swap", 0)
                            new_use_pinned_memory_for_block_swap = settings.get("use_pinned_memory_for_block_swap", False)
                            new_prec = settings.get("mixed_precision", "bf16")
                            new_grad_cp = settings.get("gradient_checkpointing", True)
                            new_fp8_s = settings.get("fp8_scaled", True)
                            new_fp8_l = settings.get("fp8_llm", False)
                            new_full_bf16 = settings.get("full_bf16", False)
                            new_fused_backward_pass = settings.get("fused_backward_pass", False)
                            new_mem_eff_save = settings.get("mem_eff_save", False)
                            new_block_swap_optimizer_patch_params = settings.get("block_swap_optimizer_patch_params", False)
                            new_add_args = settings.get("additional_args", "")
        
                            # Sample image params
                            new_sample_enable = settings.get("sample_images", False)
                            new_sample_every_n = settings.get("sample_every_n_epochs", 1)
                            new_sample_output_dir = settings.get("sample_output_dir", "")
                            new_sample_prompt = settings.get("sample_prompt", "")
                            new_sample_negative = settings.get("sample_negative_prompt", "")
                            new_sample_w = settings.get("sample_w", new_w)
                            new_sample_h = settings.get("sample_h", new_h)
        
                            # Post-processing params
                            new_in_lora = settings.get("input_lora_path", "")
                            new_out_comfy = settings.get("output_comfy_lora_path", "")
                            new_lokr_rank = settings.get("lokr_rank", "")
        
                            # Load dataset config content
                            preview_content = load_dataset_config_content(path)
        
                            msg = f"Project initialized at {path}. "
                            if settings:
                                msg += " Settings loaded."
                            msg += " 'training' folder ready. Configure the dataset in the 'training' folder. Images and caption files (same name as image, extension is '.txt') should be placed in the 'training' folder."
                            msg += "\n\nProject initialized."
                            if settings:
                                msg += " Settings loaded from musubi_project.toml."
                            msg += " Place images and matching caption files (.txt) in the 'training' folder."
        
                            updates = {
                                "model_arch": new_model,
                                "vram_size": new_vram,
                                "comfy_models_dir": new_comfy,
                                "resolution_w": new_w,
                                "resolution_h": new_h,
                                "batch_size": new_batch,
                                "control_directory": new_control_dir,
                                "control_res_w": new_control_w,
                                "control_res_h": new_control_h,
                                "no_resize_control": new_no_resize_control,
                                "image_directory": new_image_dir,
                                "cache_directory": new_cache_dir,
                                "caption_extension": new_caption_ext,
                                "num_repeats": new_num_repeats,
                                "enable_bucket": new_enable_bucket,
                                "bucket_no_upscale": new_bucket_no_upscale,
                                "toml_preview": preview_content,
                                "vae_path": new_vae,
                                "text_encoder1_path": new_te1,
                                "text_encoder2_path": new_te2,
                                "dit_path": new_dit,
                                "output_name": new_out_nm,
                                "network_dim": new_dim,
                                "learning_rate": new_lr,
                                "optimizer_type": new_optimizer_type,
                                "optimizer_args": new_optimizer_args,
                                "lr_scheduler": new_lr_scheduler,
                                "lr_scheduler_args": new_lr_scheduler_args,
                                "training_mode": new_training_mode,
                                "network_type": new_network_type,
                                "network_args": new_network_args,
                                "network_alpha": new_network_alpha,
                                "lr_warmup_steps": new_lr_warmup_steps,
                                "seed": new_seed,
                                "max_grad_norm": new_max_grad_norm,
                                "num_epochs": new_epochs,
                                "save_every_n_epochs": new_save_n,
                                "discrete_flow_shift": new_flow,
                                "block_swap": new_swap,
                                "use_pinned_memory_for_block_swap": new_use_pinned_memory_for_block_swap,
                                "mixed_precision": new_prec,
                                "gradient_checkpointing": new_grad_cp,
                                "fp8_scaled": new_fp8_s,
                                "fp8_llm": new_fp8_l,
                                "full_bf16": new_full_bf16,
                                "fused_backward_pass": new_fused_backward_pass,
                                "mem_eff_save": new_mem_eff_save,
                                "block_swap_optimizer_patch_params": new_block_swap_optimizer_patch_params,
                                "additional_args": new_add_args,
                                "sample_images": new_sample_enable,
                                "sample_every_n": new_sample_every_n,
                                "sample_output_dir": new_sample_output_dir,
                                "sample_prompt": new_sample_prompt,
                                "sample_negative_prompt": new_sample_negative,
                                "sample_w": new_sample_w,
                                "sample_h": new_sample_h,
                                "input_lora": new_in_lora,
                                "output_comfy_lora": new_out_comfy,
                                "lokr_rank": new_lokr_rank,
                            }
                            return (msg, *pack_updates(INIT_OUTPUT_KEYS, updates))
                        except Exception as e:
                            return (f"Error initializing project: {str(e)}", *pack_updates(INIT_OUTPUT_KEYS, {}))
        
                    def generate_config(
                        project_path,
                        w,
                        h,
                        batch,
                        model_val,
                        vram_val,
                        comfy_val,
                        vae_val,
                        te1_val,
                        te2_val,
                        control_dir,
                        control_w,
                        control_h,
                        no_resize_ctrl,
                        image_dir_val,
                        cache_dir_val,
                        caption_ext_val,
                        num_repeats_val,
                        enable_bucket_val,
                        bucket_no_upscale_val,
                    ):
                        if not project_path:
                            return "Error: Project directory not specified.", ""
        
                        # Save project settings first
                        save_project_settings(
                            project_path,
                            model_arch=model_val,
                            vram_size=vram_val,
                            comfy_models_dir=comfy_val,
                            resolution_w=w,
                            resolution_h=h,
                            batch_size=batch,
                            vae_path=vae_val,
                            text_encoder1_path=te1_val,
                            text_encoder2_path=te2_val,
                            control_directory=control_dir,
                            control_resolution_w=control_w,
                            control_resolution_h=control_h,
                            no_resize_control=no_resize_ctrl,
                            image_directory=image_dir_val,
                            cache_directory=cache_dir_val,
                            caption_extension=caption_ext_val,
                            num_repeats=num_repeats_val,
                            enable_bucket=enable_bucket_val,
                            bucket_no_upscale=bucket_no_upscale_val,
                        )
        
                        # Normalize paths
                        project_path = os.path.abspath(project_path)
                        image_dir_raw = (image_dir_val or "").strip()
                        cache_dir_raw = (cache_dir_val or "").strip()
                        if not image_dir_raw:
                            image_dir_raw = os.path.join(project_path, "training")
                        if not cache_dir_raw:
                            cache_dir_raw = os.path.join(project_path, "cache")
        
                        image_dir = image_dir_raw.replace("\\", "/")
                        cache_dir = cache_dir_raw.replace("\\", "/")
        
                        caption_ext = (caption_ext_val or ".txt").strip()
                        if not caption_ext:
                            caption_ext = ".txt"
                        try:
                            num_repeats_int = int(num_repeats_val)
                        except Exception:
                            num_repeats_int = 1
        
                        toml_content = textwrap.dedent(
                            f"""\
                            # Auto-generated by Musubi Tuner GUI

                            [general]
                            resolution = [{int(w)}, {int(h)}]
                            caption_extension = "{caption_ext}"
                            batch_size = {int(batch)}
                            enable_bucket = {str(bool(enable_bucket_val)).lower()}
                            bucket_no_upscale = {str(bool(bucket_no_upscale_val)).lower()}

                            [[datasets]]
                            image_directory = "{image_dir}"
                            cache_directory = "{cache_dir}"
                            num_repeats = {num_repeats_int}
                            """
                        )
                        control_dir = (control_dir or "").strip()
                        if control_dir:
                            safe_control_dir = control_dir.replace("\\", "/")
                            toml_content += f'control_directory = "{safe_control_dir}"\n'
                            if no_resize_ctrl:
                                toml_content += "no_resize_control = true\n"
                            try:
                                control_w_int = int(control_w)
                                control_h_int = int(control_h)
                            except Exception:
                                control_w_int = 0
                                control_h_int = 0
                            if control_w_int > 0 and control_h_int > 0:
                                toml_content += f"control_resolution = [{control_w_int}, {control_h_int}]\n"
                        try:
                            config_path = os.path.join(project_path, "dataset_config.toml")
                            with open(config_path, "w", encoding="utf-8") as f:
                                f.write(toml_content)
                            return f"Successfully generated config at / 險ｭ螳壹ヵ繧｡繧､繝ｫ縺御ｽ懈・縺輔ｌ縺ｾ縺励◆: {config_path}", toml_content
                        except Exception as e:
                            return f"Error generating config / 險ｭ螳壹ヵ繧｡繧､繝ｫ縺ｮ逕滓・縺ｫ螟ｱ謨励＠縺ｾ縺励◆: {str(e)}", ""
        
                    def validate_models_dir(path):
                        if not path:
                            return "Please enter a ComfyUI models directory."
        
                        required_subdirs = ["diffusion_models", "vae", "text_encoders"]
                        missing = []
                        for d in required_subdirs:
                            if not os.path.exists(os.path.join(path, d)):
                                missing.append(d)
        
                        if missing:
                            return f"Error: Missing subdirectories in models folder / models繝輔か繝ｫ繝縺ｫ莉･荳九・繧ｵ繝悶ョ繧｣繝ｬ繧ｯ繝医Μ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: {', '.join(missing)}"
        
                        return "Valid ComfyUI models directory structure found."
        
                    def set_recommended_settings(project_path, model_arch, vram_val):
                        model_arch = _normalize_model_label(model_arch)
                        w, h = config_manager.get_resolution(model_arch)
                        recommended_batch_size = config_manager.get_batch_size(model_arch, vram_val)
        
                        if project_path:
                            save_project_settings(project_path, resolution_w=w, resolution_h=h, batch_size=recommended_batch_size)
                        return w, h, recommended_batch_size
        
                    def set_preprocessing_defaults(project_path, comfy_models_dir, model_arch):
                        model_arch = _normalize_model_label(model_arch)
                        if not comfy_models_dir:
                            return gr.update(), gr.update(), gr.update()
        
                        vae_default, te1_default, te2_default = config_manager.get_preprocessing_paths(model_arch, comfy_models_dir)
                        if not te2_default:
                            te2_default = ""  # Ensure empty string for text input
                        if model_arch.startswith("Flux.2") and te1_default and not os.path.exists(te1_default):
                            # Flux.2 text encoder weights are often split; avoid auto-filling a bad path.
                            te1_default = ""
        
                        if project_path:
                            save_project_settings(
                                project_path, vae_path=vae_default, text_encoder1_path=te1_default, text_encoder2_path=te2_default
                            )
        
                        return vae_default, te1_default, te2_default
        
                    def auto_detect_paths(project_path, comfy_models_dir, model_arch):
                        model_arch = _normalize_model_label(model_arch)
                        if not comfy_models_dir:
                            return gr.update(), gr.update(), gr.update(), gr.update(), i18n("msg_auto_detect_fail").format(
                                e="ComfyUI models directory not set"
                            )
        
                        base_dir = os.path.abspath(comfy_models_dir)
                        diffusion_dir = os.path.join(base_dir, "diffusion_models")
                        vae_dir = os.path.join(base_dir, "vae")
                        te_dir = os.path.join(base_dir, "text_encoders")
        
                        def find_first(search_dir, patterns):
                            if not search_dir or not os.path.exists(search_dir):
                                return ""
                            for pattern in patterns:
                                candidates = [
                                    os.path.join(search_dir, pattern),
                                    os.path.join(search_dir, "**", pattern),
                                ]
                                for candidate in candidates:
                                    matches = glob.glob(candidate, recursive=True)
                                    matches.sort()
                                    if matches:
                                        return matches[0]
                            return ""
        
                        if model_arch == "Flux.2 Dev":
                            dit_patterns = [
                                "flux2-dev.safetensors",
                                "flux2_dev.safetensors",
                                "flux-2-dev.safetensors",
                                "flux_2_dev.safetensors",
                                "*flux2*dev*.safetensors",
                                "*flux-2*dev*.safetensors",
                                "*flux_2*dev*.safetensors",
                            ]
                            vae_patterns = ["ae.safetensors"]
                            te_patterns = [
                                "*mistral*00001-of-*.safetensors",
                                "*mistral*00001*.safetensors",
                                "*mistral*.safetensors",
                            ]
                        elif model_arch == "Flux.2 Klein (4B)":
                            dit_patterns = [
                                "flux2-klein-4b.safetensors",
                                "flux2_klein_4b.safetensors",
                                "flux-2-klein-4b.safetensors",
                                "flux_2_klein_4b.safetensors",
                                "*flux2*klein*4b*.safetensors",
                                "*flux-2*klein*4b*.safetensors",
                                "*flux_2*klein*4b*.safetensors",
                            ]
                            vae_patterns = ["ae.safetensors"]
                            te_patterns = [
                                "qwen_3_4b.safetensors",
                                "*qwen*3*4b*00001-of-*.safetensors",
                                "*qwen*3*4b*.safetensors",
                                "*qwen*4b*00001-of-*.safetensors",
                                "*qwen*4b*.safetensors",
                            ]
                        elif model_arch == "Flux.2 Klein Base (4B)":
                            dit_patterns = [
                                "flux2-klein-base-4b.safetensors",
                                "flux2_klein_base_4b.safetensors",
                                "flux-2-klein-base-4b.safetensors",
                                "flux_2_klein_base_4b.safetensors",
                                "*flux2*klein*base*4b*.safetensors",
                                "*flux-2*klein*base*4b*.safetensors",
                                "*flux_2*klein*base*4b*.safetensors",
                            ]
                            vae_patterns = ["ae.safetensors"]
                            te_patterns = [
                                "qwen_3_4b.safetensors",
                                "*qwen*3*4b*00001-of-*.safetensors",
                                "*qwen*3*4b*.safetensors",
                                "*qwen*4b*00001-of-*.safetensors",
                                "*qwen*4b*.safetensors",
                            ]
                        elif model_arch == "Qwen-Image":
                            dit_patterns = ["qwen_image_bf16.safetensors", "*qwen*image*bf16*.safetensors"]
                            vae_patterns = ["qwen_image_vae.safetensors", "*qwen*image*vae*.safetensors"]
                            te_patterns = ["qwen_2.5_vl_7b.safetensors", "*qwen*2.5*vl*7b*.safetensors", "*qwen*vl*7b*.safetensors"]
                        else:  # Z-Image (default)
                            dit_patterns = [
                                "z_image_de_turbo_v1_bf16.safetensors",
                                "*z*image*de*turbo*bf16*.safetensors",
                                "*z*image*de*turbo*.safetensors",
                            ]
                            vae_patterns = ["ae.safetensors"]
                            te_patterns = ["qwen_3_4b.safetensors", "*qwen*3*4b*.safetensors"]
        
                        vae_found = find_first(vae_dir, vae_patterns)
                        te1_found = find_first(te_dir, te_patterns)
                        dit_found = find_first(diffusion_dir, dit_patterns)
        
                        # Save settings if project path is provided
                        if project_path:
                            updates = {}
                            if vae_found:
                                updates["vae_path"] = vae_found
                            if te1_found:
                                updates["text_encoder1_path"] = te1_found
                            if dit_found:
                                updates["dit_path"] = dit_found
                            if updates:
                                save_project_settings(project_path, **updates)
        
                        # Build status message
                        label_vae = i18n("lbl_vae_path")
                        label_te1 = i18n("lbl_te1_path")
                        label_dit = i18n("lbl_dit_path")
        
                        found_lines = []
                        missing_labels = []
                        if vae_found:
                            found_lines.append(f"- **{label_vae}**: `{vae_found}`")
                        else:
                            missing_labels.append(label_vae)
                        if te1_found:
                            found_lines.append(f"- **{label_te1}**: `{te1_found}`")
                        else:
                            missing_labels.append(label_te1)
                        if dit_found:
                            found_lines.append(f"- **{label_dit}**: `{dit_found}`")
                        else:
                            missing_labels.append(label_dit)
        
                        lines = [f"**{i18n('msg_auto_detect_title')}**"]
                        if found_lines:
                            lines.append(f"{i18n('msg_auto_detect_found')}:")
                            lines.extend(found_lines)
                        if missing_labels:
                            lines.append(f"{i18n('msg_auto_detect_missing')}: {', '.join(missing_labels)}")
                        if model_arch.startswith("Flux.2") and not te1_found:
                            lines.append(i18n("msg_auto_detect_note_split"))
        
                        status_msg = "\n".join(lines)
        
                        return (
                            gr.update(value=vae_found) if vae_found else gr.update(),
                            gr.update(value=te1_found) if te1_found else gr.update(),
                            gr.update(),
                            gr.update(value=dit_found) if dit_found else gr.update(),
                            status_msg,
                        )
        
                    def set_training_defaults(project_path, comfy_models_dir, model_arch, vram_val):
                        model_arch = _normalize_model_label(model_arch)
                        # Get number of images from project_path to adjust num_epochs later
                        cache_dir = os.path.join(project_path, "cache")
                        if model_arch == "Qwen-Image":
                            arch_tag = "qwen_image"
                        elif model_arch.startswith("Flux.2"):
                            arch_tag = "flux_2"
                        else:
                            arch_tag = "z_image"
                        if os.path.exists(cache_dir):
                            all_files = glob.glob(os.path.join(cache_dir, f"*_{arch_tag}*.safetensors"))
                            latent_files = [f for f in all_files if not f.endswith("_te.safetensors")]
                            num_images = len(latent_files)
                        else:
                            num_images = 0
        
                        # Get training defaults from config manager
                        defaults = config_manager.get_training_defaults(model_arch, vram_val, comfy_models_dir)
        
                        # Adjust num_epochs based on number of images (simple heuristic)
                        default_num_steps = defaults.get("default_num_steps", 1000)
                        if num_images > 0:
                            adjusted_epochs = max(1, int((default_num_steps / num_images)))
                        else:
                            adjusted_epochs = 16  # Fallback default
                        sample_every_n_epochs = (adjusted_epochs // 4) if adjusted_epochs >= 4 else 1
        
                        dit_default = defaults.get("dit_path", "")
                        dim = defaults.get("network_dim", 4)
                        lr = defaults.get("learning_rate", 1e-4)
                        optimizer_type = defaults.get("optimizer_type", "adamw8bit")
                        optimizer_args = defaults.get("optimizer_args", "")
                        training_mode = defaults.get("training_mode", TRAINING_MODE_LORA)
                        if not _is_zimage(model_arch):
                            training_mode = TRAINING_MODE_LORA
                        network_type = defaults.get("network_type", "LoRA")
                        network_args = defaults.get("network_args", "")
                        network_alpha = defaults.get("network_alpha", 1)
                        lr_warmup_steps = defaults.get("lr_warmup_steps", 0)
                        seed = defaults.get("seed", 42)
                        max_grad_norm = defaults.get("max_grad_norm", 1.0)
                        epochs = adjusted_epochs
                        save_n = defaults.get("save_every_n_epochs", 1)
                        flow = defaults.get("discrete_flow_shift", 2.0)
                        swap = defaults.get("block_swap", 0)
                        use_pinned_memory_for_block_swap = defaults.get("use_pinned_memory_for_block_swap", False)
                        prec = defaults.get("mixed_precision", "bf16")
                        grad_cp = defaults.get("gradient_checkpointing", True)
                        fp8_s = defaults.get("fp8_scaled", True)
                        fp8_l = defaults.get("fp8_llm", False)
                        full_bf16 = defaults.get("full_bf16", False)
                        fused_backward_pass = defaults.get("fused_backward_pass", False)
                        mem_eff_save = defaults.get("mem_eff_save", False)
                        block_swap_optimizer_patch_params = defaults.get("block_swap_optimizer_patch_params", False)

                        sample_w = config_manager.get_resolution(model_arch)[0]
                        sample_h = config_manager.get_resolution(model_arch)[1]
                        sample_out = ""
        
                        if project_path:
                            save_project_settings(
                                project_path,
                                dit_path=dit_default,
                                network_dim=dim,
                                learning_rate=lr,
                                optimizer_type=optimizer_type,
                                optimizer_args=optimizer_args,
                                training_mode=training_mode,
                                network_type=network_type,
                                network_args=network_args,
                                network_alpha=network_alpha,
                                lr_warmup_steps=lr_warmup_steps,
                                seed=seed,
                                max_grad_norm=max_grad_norm,
                                num_epochs=epochs,
                                save_every_n_epochs=save_n,
                                discrete_flow_shift=flow,
                                block_swap=swap,
                                use_pinned_memory_for_block_swap=use_pinned_memory_for_block_swap,
                                mixed_precision=prec,
                                gradient_checkpointing=grad_cp,
                                fp8_scaled=fp8_s,
                                fp8_llm=fp8_l,
                                full_bf16=full_bf16,
                                fused_backward_pass=fused_backward_pass,
                                mem_eff_save=mem_eff_save,
                                block_swap_optimizer_patch_params=block_swap_optimizer_patch_params,
                                sample_every_n_epochs=sample_every_n_epochs,
                                sample_w=sample_w,
                                sample_h=sample_h,
                                sample_output_dir=sample_out,
                            )
        
                        updates = {
                            "dit_path": dit_default,
                            "network_dim": dim,
                            "learning_rate": lr,
                            "optimizer_type": optimizer_type,
                            "optimizer_args": optimizer_args,
                            "training_mode": training_mode,
                            "network_type": network_type,
                            "network_args": network_args,
                            "network_alpha": network_alpha,
                            "lr_warmup_steps": lr_warmup_steps,
                            "seed": seed,
                            "max_grad_norm": max_grad_norm,
                            "num_epochs": epochs,
                            "save_every_n_epochs": save_n,
                            "discrete_flow_shift": flow,
                            "block_swap": swap,
                            "use_pinned_memory_for_block_swap": use_pinned_memory_for_block_swap,
                            "mixed_precision": prec,
                            "gradient_checkpointing": grad_cp,
                            "fp8_scaled": fp8_s,
                            "fp8_llm": fp8_l,
                            "full_bf16": full_bf16,
                            "fused_backward_pass": fused_backward_pass,
                            "mem_eff_save": mem_eff_save,
                            "block_swap_optimizer_patch_params": block_swap_optimizer_patch_params,
                            "sample_every_n": sample_every_n_epochs,
                            "sample_w": sample_w,
                            "sample_h": sample_h,
                            "sample_output_dir": sample_out,
                        }
                        return pack_updates(TRAINING_DEFAULT_KEYS, updates)
        
                    def estimate_vram(model_arch, w, h, batch, prec, grad_cp, fp8_s, fp8_l, swap):
                        model_arch = _normalize_model_label(model_arch)
                        try:
                            w = int(w)
                            h = int(h)
                            batch = int(batch)
                        except Exception:
                            return i18n("msg_est_vram_note")
        
                        base_map = {
                            "Z-Image": 8.0,
                            "Qwen-Image": 18.0,
                            "Flux.2 Dev": 18.0,
                            "Flux.2 Klein (4B)": 12.0,
                            "Flux.2 Klein Base (4B)": 14.0,
                        }
                        base_vram = base_map.get(model_arch, 12.0)
        
                        scale = (w * h) / (1024 * 1024) * max(1, batch)
                        vram = base_vram * scale
        
                        if prec == "no":
                            vram *= 1.6
                        elif prec == "fp16":
                            vram *= 0.95
        
                        if grad_cp:
                            vram *= 0.85
        
                        if fp8_s:
                            vram *= 0.85
        
                        if fp8_l:
                            vram *= 0.9
        
                        try:
                            swap_val = int(swap)
                        except Exception:
                            swap_val = 0
        
                        if swap_val > 0:
                            max_swap_map = {
                                "Z-Image": 28,
                                "Qwen-Image": 58,
                                "Flux.2 Dev": 29,
                                "Flux.2 Klein (4B)": 13,
                                "Flux.2 Klein Base (4B)": 13,
                            }
                            max_swap = max_swap_map.get(model_arch, 30)
                            swap_ratio = min(1.0, swap_val / max_swap)
                            vram *= (1.0 - 0.4 * swap_ratio)
        
                        vram = max(1.0, vram)
                        low = vram * 0.8
                        high = vram * 1.2
        
                        return (
                            f"**{i18n('msg_est_vram_title')}**: ~{low:.1f} - {high:.1f} GB\n\n"
                            f"{i18n('msg_est_vram_note')}"
                        )
        
                    def _prefer_auto(auto_val, fallback):
                        if isinstance(auto_val, dict):
                            if auto_val.get("value"):
                                return auto_val
                            return fallback
                        if isinstance(auto_val, str) and auto_val:
                            return auto_val
                        return fallback
        
                    def quick_setup(project_path, model_arch, vram_val, comfy_models_dir):
                        w, h, batch = set_recommended_settings(project_path, model_arch, vram_val)
        
                        vae_default, te1_default, te2_default = set_preprocessing_defaults(project_path, comfy_models_dir, model_arch)
                        (
                            dit_default,
                            dim,
                            lr,
                            optimizer_type,
                            optimizer_args,
                            training_mode,
                            network_type,
                            network_args,
                            network_alpha,
                            lr_warmup_steps,
                            seed,
                            max_grad_norm,
                            epochs,
                            save_n,
                            flow,
                            swap,
                            use_pinned_memory_for_block_swap,
                            prec,
                            grad_cp,
                            fp8_s,
                            fp8_l,
                            full_bf16,
                            fused_backward_pass,
                            mem_eff_save,
                            block_swap_optimizer_patch_params,
                            sample_every_n_epochs,
                            sample_w_default,
                            sample_h_default,
                            sample_out,
                        ) = set_training_defaults(project_path, comfy_models_dir, model_arch, vram_val)
        
                        auto_vae = gr.update()
                        auto_te1 = gr.update()
                        auto_te2 = gr.update()
                        auto_dit = gr.update()
                        auto_status = ""
                        if comfy_models_dir:
                            auto_vae, auto_te1, auto_te2, auto_dit, auto_status = auto_detect_paths(
                                project_path, comfy_models_dir, model_arch
                            )
        
                        status_msg = i18n("msg_quick_setup_done")
                        if auto_status:
                            status_msg = f"{status_msg}\n\n{auto_status}"
        
                        updates = {
                            "resolution_w": w,
                            "resolution_h": h,
                            "batch_size": batch,
                            "vae_path": _prefer_auto(auto_vae, vae_default),
                            "text_encoder1_path": _prefer_auto(auto_te1, te1_default),
                            "text_encoder2_path": _prefer_auto(auto_te2, te2_default),
                            "dit_path": _prefer_auto(auto_dit, dit_default),
                            "network_dim": dim,
                            "learning_rate": lr,
                            "optimizer_type": optimizer_type,
                            "optimizer_args": optimizer_args,
                            "training_mode": training_mode,
                            "network_type": network_type,
                            "network_args": network_args,
                            "network_alpha": network_alpha,
                            "lr_warmup_steps": lr_warmup_steps,
                            "seed": seed,
                            "max_grad_norm": max_grad_norm,
                            "num_epochs": epochs,
                            "save_every_n_epochs": save_n,
                            "discrete_flow_shift": flow,
                            "block_swap": swap,
                            "use_pinned_memory_for_block_swap": use_pinned_memory_for_block_swap,
                            "mixed_precision": prec,
                            "gradient_checkpointing": grad_cp,
                            "fp8_scaled": fp8_s,
                            "fp8_llm": fp8_l,
                            "full_bf16": full_bf16,
                            "fused_backward_pass": fused_backward_pass,
                            "mem_eff_save": mem_eff_save,
                            "block_swap_optimizer_patch_params": block_swap_optimizer_patch_params,
                            "sample_every_n": sample_every_n_epochs,
                            "sample_w": sample_w_default,
                            "sample_h": sample_h_default,
                            "sample_output_dir": sample_out,
                            "training_model_info": update_model_info(model_arch),
                            "quick_status": status_msg,
                        }
                        return pack_updates(QUICK_SETUP_OUTPUT_KEYS, updates)
        
                    def check_missing(
                        project_path,
                        comfy_models_dir,
                        vae_val,
                        te1_val,
                        dit_val,
                    ):
                        missing = []
                        if not project_path:
                            missing.append(i18n("lbl_proj_dir"))
                        else:
                            config_path = os.path.join(project_path, "dataset_config.toml")
                            if not os.path.exists(config_path):
                                missing.append("dataset_config.toml")
                        if not comfy_models_dir:
                            missing.append(i18n("lbl_comfy_dir"))
                        if not vae_val:
                            missing.append(i18n("lbl_vae_path"))
                        if not te1_val:
                            missing.append(i18n("lbl_te1_path"))
                        if not dit_val:
                            missing.append(i18n("lbl_dit_path"))
        
                        if missing:
                            return f"**{i18n('msg_missing_title')}**\n- " + "\n- ".join(missing)
                        return "OK"
        
                    def set_post_processing_defaults(project_path, output_nm):
                        if not project_path or not output_nm:
                            return gr.update(), gr.update()
        
                        models_dir = os.path.join(project_path, "models")
                        in_lora = os.path.join(models_dir, f"{output_nm}.safetensors")
                        out_lora = os.path.join(models_dir, f"{output_nm}_comfy.safetensors")
        
                        save_project_settings(project_path, input_lora_path=in_lora, output_comfy_lora_path=out_lora)
        
                        return in_lora, out_lora
        
                    import subprocess
                    import sys
        
                    def run_command(command):
                        try:
                            encoding = locale.getpreferredencoding(False) or "utf-8"
                            process = subprocess.Popen(
                                command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                shell=True,
                                text=True,
                                encoding=encoding,
                                errors="replace",
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                            )
        
                            output_log = command + "\n\n"
                            for line in process.stdout:
                                output_log += line
                                yield output_log
        
                            process.wait()
                            if process.returncode != 0:
                                output_log += (
                                    f"\nError: Process exited with code / 繝励Ο繧ｻ繧ｹ縺梧ｬ｡縺ｮ繧ｳ繝ｼ繝峨〒繧ｨ繝ｩ繝ｼ邨ゆｺ・＠縺ｾ縺励◆: {process.returncode}"
                                )
                                yield output_log
                            else:
                                output_log += "\nProcess completed successfully / 繝励Ο繧ｻ繧ｹ縺梧ｭ｣蟶ｸ縺ｫ螳御ｺ・＠縺ｾ縺励◆"
                                yield output_log
        
                        except Exception as e:
                            yield f"Error executing command / 繧ｳ繝槭Φ繝峨・螳溯｡御ｸｭ縺ｫ繧ｨ繝ｩ繝ｼ縺檎匱逕溘＠縺ｾ縺励◆: {str(e)}"
        
                    def cache_latents(project_path, vae_path_val, te1, te2, model, comfy, w, h, batch, vram_val):
                        if not project_path:
                            yield "Error: Project directory not set."
                            return
                        model = _normalize_model_label(model)
        
                        # Save settings first
                        save_project_settings(
                            project_path,
                            model_arch=model,
                            comfy_models_dir=comfy,
                            resolution_w=w,
                            resolution_h=h,
                            batch_size=batch,
                            vae_path=vae_path_val,
                            text_encoder1_path=te1,
                            text_encoder2_path=te2,
                        )
        
                        if not vae_path_val:
                            yield "Error: VAE path not set."
                            return
        
                        if not os.path.exists(vae_path_val):
                            yield f"Error: VAE model not found at / 謖・ｮ壹＆繧後◆繝代せ縺ｫVAE繝｢繝・Ν縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: {vae_path_val}"
                            return
        
                        config_path = os.path.join(project_path, "dataset_config.toml")
                        if not os.path.exists(config_path):
                            yield f"Error: dataset_config.toml not found in {project_path}. Please generate it first."
                            return
        
                        script_name = "zimage_cache_latents.py"
                        if model == "Qwen-Image":
                            script_name = "qwen_image_cache_latents.py"
                        elif model.startswith("Flux.2"):
                             script_name = "flux_2_cache_latents.py"
        
                        script_path = os.path.join("src", "musubi_tuner", script_name)
        
                        cmd = [sys.executable, script_path, "--dataset_config", config_path, "--vae", vae_path_val]
        
                        # Placeholder for argument modification
                        if _is_zimage(model):
                            pass
                        elif model == "Qwen-Image":
                            pass
                        elif model == "Flux.2 Dev":
                            cmd.extend(["--model_version", "dev"])
                        elif model == "Flux.2 Klein (4B)":
                            cmd.extend(["--model_version", "klein-4b"])
                        elif model == "Flux.2 Klein Base (4B)":
                            cmd.extend(["--model_version", "klein-base-4b"])
        
                        command_str = " ".join(cmd)
                        yield f"Starting Latent Caching. Please wait for the first log to appear. / Latent縺ｮ繧ｭ繝｣繝・す繝･繧帝幕蟋九＠縺ｾ縺吶よ怙蛻昴・繝ｭ繧ｰ縺瑚｡ｨ遉ｺ縺輔ｌ繧九∪縺ｧ縺ｫ縺励・繧峨￥縺九°繧翫∪縺吶・nCommand: {command_str}\n\n"
        
                        yield from run_command(command_str)
        
                    def cache_text_encoder(
                        project_path, te1_path_val, te2_path_val, vae, model, comfy, w, h, batch, vram_val, fp8_text_encoder
                    ):
                        if not project_path:
                            yield "Error: Project directory not set."
                            return
                        model = _normalize_model_label(model)
        
                        # Save settings first
                        save_project_settings(
                            project_path,
                            model_arch=model,
                            comfy_models_dir=comfy,
                            resolution_w=w,
                            resolution_h=h,
                            batch_size=batch,
                            vae_path=vae,
                            text_encoder1_path=te1_path_val,
                            text_encoder2_path=te2_path_val,
                        )
        
                        if not te1_path_val:
                            yield "Error: Text Encoder 1 path not set."
                            return
        
                        if not os.path.exists(te1_path_val):
                            yield f"Error: Text Encoder 1 model not found at / 謖・ｮ壹＆繧後◆繝代せ縺ｫText Encoder 1繝｢繝・Ν縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: {te1_path_val}"
                            return
        
                        # Z-Image only uses te1 for now, but keeping te2 in signature if needed later or for other models
        
                        config_path = os.path.join(project_path, "dataset_config.toml")
                        if not os.path.exists(config_path):
                            yield f"Error: dataset_config.toml not found in {project_path}. Please generate it first."
                            return
        
                        script_name = "zimage_cache_text_encoder_outputs.py"
                        if model == "Qwen-Image":
                            script_name = "qwen_image_cache_text_encoder_outputs.py"
                        elif model.startswith("Flux.2"):
                            script_name = "flux_2_cache_text_encoder_outputs.py"
        
                        script_path = os.path.join("src", "musubi_tuner", script_name)
        
                        cmd = [
                            sys.executable,
                            script_path,
                            "--dataset_config",
                            config_path,
                            "--text_encoder",
                            te1_path_val,
                            "--batch_size",
                            "1",  # Conservative default
                        ]
        
                        # Model-specific argument modification
                        if _is_zimage(model):
                            pass
                        elif model == "Qwen-Image":
                            # Add --fp8_vl for low VRAM (16GB or less)
                            if vram_val in ["12", "16"]:
                                cmd.append("--fp8_vl")
                        elif model == "Flux.2 Dev":
                            cmd.extend(["--model_version", "dev"])
                        elif model == "Flux.2 Klein (4B)":
                            cmd.extend(["--model_version", "klein-4b"])
                        elif model == "Flux.2 Klein Base (4B)":
                            cmd.extend(["--model_version", "klein-base-4b"])
        
                        if model.startswith("Flux.2") and model != "Flux.2 Dev" and fp8_text_encoder:
                            cmd.append("--fp8_text_encoder")
        
                        command_str = " ".join(cmd)
                        yield f"Starting Text Encoder Caching. Please wait for the first log to appear. / Text Encoder縺ｮ繧ｭ繝｣繝・す繝･繧帝幕蟋九＠縺ｾ縺吶よ怙蛻昴・繝ｭ繧ｰ縺瑚｡ｨ遉ｺ縺輔ｌ繧九∪縺ｧ縺ｫ縺励・繧峨￥縺九°繧翫∪縺吶・nCommand: {command_str}\n\n"
        
                        yield from run_command(command_str)
        
            with gr.TabItem(tab_label(i18n("acc_preprocessing"))):
                with gr.Group(elem_classes=["section-card", "card-preprocess"]):
                    gr.Markdown(i18n("acc_preprocessing"))
                    gr.Markdown(i18n("desc_preprocessing"))
                    with gr.Row():
                        set_preprocessing_defaults_btn = gr.Button(i18n("btn_set_paths"))
                        auto_detect_paths_btn = gr.Button(i18n("btn_auto_detect_paths"))
                    auto_detect_status = gr.Markdown("")
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        vae_path = gr.Textbox(label=i18n("lbl_vae_path"), placeholder=i18n("ph_vae_path"), max_lines=1, scale=8)
                        browse_vae = gr.Button(i18n("btn_browse"), scale=1)
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        text_encoder1_path = gr.Textbox(label=i18n("lbl_te1_path"), placeholder=i18n("ph_te1_path"), max_lines=1, scale=8)
                        browse_te1 = gr.Button(i18n("btn_browse"), scale=1)
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        text_encoder2_path = gr.Textbox(label=i18n("lbl_te2_path"), placeholder=i18n("ph_te2_path"), max_lines=1, scale=8)
                        browse_te2 = gr.Button(i18n("btn_browse"), scale=1)
        
                    with gr.Row():
                        cache_latents_btn = gr.Button(i18n("btn_cache_latents"))
                        cache_text_btn = gr.Button(i18n("btn_cache_text"))
        
                    # Simple output area for caching logs
                    caching_output = gr.Textbox(label=i18n("lbl_cache_log"), lines=10, interactive=False)
        
            with gr.TabItem(tab_label(i18n("acc_training"))):
                with gr.Group(elem_classes=["section-card", "card-training"]):
                    gr.Markdown(i18n("acc_training"))
                    gr.Markdown(i18n("desc_training_basic"))
                    training_model_info = gr.Markdown(i18n("desc_training_flux2"))
        
                    with gr.Row():
                        set_training_defaults_btn = gr.Button(i18n("btn_rec_params"))
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        dit_path = gr.Textbox(label=i18n("lbl_dit_path"), placeholder=i18n("ph_dit_path"), max_lines=1, scale=8)
                        browse_dit = gr.Button(i18n("btn_browse"), scale=1)
        
                    with gr.Row():
                        output_name = gr.Textbox(label=i18n("lbl_output_name"), value="my_lora", max_lines=1)
        
                    with gr.Group():
                        gr.Markdown(i18n("header_basic_params"))
                        with gr.Row():
                            training_mode = gr.Dropdown(
                                label=i18n("lbl_training_mode"),
                                choices=[TRAINING_MODE_LORA],
                                value=TRAINING_MODE_LORA,
                            )
                            network_type = gr.Dropdown(
                                label=i18n("lbl_network_type"),
                                choices=["LoRA", "LoHa", "LoKr"],
                                value="LoRA",
                            )
                            network_dim = gr.Number(label=i18n("lbl_dim"), value=4)
                            network_alpha = gr.Number(label=i18n("lbl_network_alpha"), value=1)
                        with gr.Row():
                            network_args = gr.Textbox(
                                label=i18n("lbl_network_args"),
                                placeholder="e.g. rank_dropout=0.1 module_dropout=0.1",
                            )
                        with gr.Row(visible=False) as finetune_opts_row:
                            full_bf16 = gr.Checkbox(label=i18n("lbl_full_bf16"), value=False)
                            fused_backward_pass = gr.Checkbox(label=i18n("lbl_fused_backward_pass"), value=False)
                            mem_eff_save = gr.Checkbox(label=i18n("lbl_mem_eff_save"), value=False)
                            block_swap_optimizer_patch_params = gr.Checkbox(
                                label=i18n("lbl_block_swap_optimizer_patch_params"), value=False
                            )
                        with gr.Row():
                            learning_rate = gr.Number(label=i18n("lbl_lr"), value=1e-4)
                            optimizer_type = gr.Dropdown(
                                label=i18n("lbl_optimizer"),
                                choices=["adamw8bit", "adamw", "adafactor"],
                                value="adamw8bit",
                            )
                            optimizer_args = gr.Textbox(
                                label=i18n("lbl_optimizer_args"), placeholder="e.g. weight_decay=0.01"
                            )
                            lr_scheduler = gr.Dropdown(
                                label=i18n("lbl_lr_scheduler"),
                                choices=[
                                    "constant",
                                    "linear",
                                    "cosine",
                                    "cosine_with_restarts",
                                    "polynomial",
                                    "constant_with_warmup",
                                    "adafactor",
                                    "rex",
                                ],
                                value="constant",
                            )
                            lr_scheduler_args = gr.Textbox(
                                label=i18n("lbl_lr_scheduler_args"), placeholder="e.g. num_cycles=1 power=1.0"
                            )
                        with gr.Row():
                            lr_warmup_steps = gr.Number(label=i18n("lbl_lr_warmup_steps"), value=0, precision=0)
                            seed = gr.Number(label=i18n("lbl_seed"), value=42, precision=0)
                            max_grad_norm = gr.Number(label=i18n("lbl_max_grad_norm"), value=1.0)
                            num_epochs = gr.Number(label=i18n("lbl_epochs"), value=16)
                            save_every_n_epochs = gr.Number(label=i18n("lbl_save_every"), value=1)
        
                    with gr.Group():
                        with gr.Row():
                            discrete_flow_shift = gr.Number(label=i18n("lbl_flow_shift"), value=1.0)
                            block_swap = gr.Slider(label=i18n("lbl_block_swap"), minimum=0, maximum=60, step=1, value=0)
                            use_pinned_memory_for_block_swap = gr.Checkbox(
                                label=i18n("lbl_use_pinned_memory_for_block_swap"),
                                value=False,
                            )
        
                        with gr.Accordion(i18n("accordion_advanced"), open=False):
                            gr.Markdown(i18n("desc_training_detailed"))
        
                        with gr.Row():
                            mixed_precision = gr.Dropdown(label=i18n("lbl_mixed_precision"), choices=["bf16", "fp16", "no"], value="bf16")
                            gradient_checkpointing = gr.Checkbox(label=i18n("lbl_grad_cp"), value=True)
        
                        with gr.Row():
                            fp8_scaled = gr.Checkbox(label=i18n("lbl_fp8_scaled"), value=True)
                            fp8_llm = gr.Checkbox(label=i18n("lbl_fp8_llm"), value=initial_fp8_llm)
        
                    with gr.Row():
                        est_vram_btn = gr.Button(i18n("btn_est_vram"))
                    vram_estimate = gr.Markdown("")
        
                    with gr.Group():
                        gr.Markdown(i18n("header_sample_images"))
                        sample_images = gr.Checkbox(label=i18n("lbl_enable_sample"), value=False)
                        with gr.Row():
                            sample_prompt = gr.Textbox(label=i18n("lbl_sample_prompt"), placeholder=i18n("ph_sample_prompt"))
                        with gr.Row():
                            sample_negative_prompt = gr.Textbox(
                                label=i18n("lbl_sample_negative_prompt"),
                                placeholder=i18n("ph_sample_negative_prompt"),
                            )
                        with gr.Row():
                            sample_w = gr.Number(label=i18n("lbl_sample_w"), value=1024, precision=0)
                            sample_h = gr.Number(label=i18n("lbl_sample_h"), value=1024, precision=0)
                            sample_every_n = gr.Number(label=i18n("lbl_sample_every_n"), value=1, precision=0)
                        with gr.Row(elem_classes=["path-row", "env-row"]):
                            sample_output_dir = gr.Textbox(
                                label=i18n("lbl_sample_output_dir"),
                                placeholder=i18n("ph_sample_output_dir"),
                                scale=8,
                            )
                            browse_sample_out = gr.Button(i18n("btn_browse"), scale=1)
        
                    with gr.Accordion(i18n("accordion_additional"), open=False):
                        gr.Markdown(i18n("desc_additional_args"))
                        additional_args = gr.Textbox(label=i18n("lbl_additional_args"), placeholder=i18n("ph_additional_args"))
        
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        resume_path = gr.Textbox(label=i18n("lbl_resume"), placeholder=i18n("ph_resume"), scale=8)
                        browse_resume = gr.Button(i18n("btn_browse"), scale=1)
        
                    training_status = gr.Markdown("")
                    with gr.Row():
                        start_training_btn = gr.Button(i18n("btn_start_training"), variant="primary", scale=2)
                        tensorboard_btn = gr.Button(i18n("btn_tensorboard"), scale=1)
        
            with gr.TabItem(tab_label(i18n("acc_post_processing"))):
                with gr.Group(elem_classes=["section-card", "card-post"]):
                    gr.Markdown(i18n("acc_post_processing"))
                    gr.Markdown(i18n("desc_post_proc"))
                    with gr.Row():
                        set_post_proc_defaults_btn = gr.Button(i18n("btn_set_paths"))
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        input_lora = gr.Textbox(label=i18n("lbl_input_lora"), placeholder=i18n("ph_input_lora"), max_lines=1, scale=8)
                        browse_input_lora = gr.Button(i18n("btn_browse"), scale=1)
                    with gr.Row(elem_classes=["path-row", "env-row"]):
                        output_comfy_lora = gr.Textbox(label=i18n("lbl_output_comfy"), placeholder=i18n("ph_output_comfy"), max_lines=1, scale=8)
                        browse_output_lora = gr.Button(i18n("btn_browse"), scale=1)
                    with gr.Row():
                        lokr_rank = gr.Textbox(label=i18n("lbl_lokr_rank"), placeholder="Optional max rank for LoKr QKV conversion")
        
                    convert_btn = gr.Button(i18n("btn_convert"))
                    conversion_log = gr.Textbox(label=i18n("lbl_conversion_log"), lines=5, interactive=False)
        
        def convert_lora_to_comfy(project_path, input_path, output_path, model, comfy, w, h, batch, vae, te1, te2, lokr_rank_val):
            if not project_path:
                yield "Error: Project directory not set."
                return

            # Save settings
            save_project_settings(
                project_path,
                model_arch=model,
                comfy_models_dir=comfy,
                resolution_w=w,
                resolution_h=h,
                batch_size=batch,
                vae_path=vae,
                text_encoder1_path=te1,
                text_encoder2_path=te2,
                input_lora_path=input_path,
                output_comfy_lora_path=output_path,
                lokr_rank=lokr_rank_val,
            )

            if not input_path or not output_path:
                yield "Error: Input and Output paths must be specified."
                return

            if not os.path.exists(input_path):
                yield f"Error: Input file not found at {input_path} / 蜈･蜉帙ヵ繧｡繧､繝ｫ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: {input_path}"
                return

            # Script path
            script_path = os.path.join("src", "musubi_tuner", "networks", "convert_z_image_lora_to_comfy.py")
            if not os.path.exists(script_path):
                yield f"Error: Conversion script not found at {script_path} / 螟画鋤繧ｹ繧ｯ繝ｪ繝励ヨ縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ: {script_path}"
                return

            cmd = [sys.executable, script_path, input_path, output_path]
            lokr_rank_text = str(lokr_rank_val).strip() if lokr_rank_val is not None else ""
            if lokr_rank_text:
                try:
                    lokr_rank_int = int(lokr_rank_text)
                    if lokr_rank_int <= 0:
                        raise ValueError("not positive")
                except Exception:
                    yield "Error: LoKr rank must be a positive integer when specified."
                    return
                cmd.extend(["--lokr_rank", str(lokr_rank_int)])

            command_str = " ".join(cmd)
            yield f"Starting Conversion. / 螟画鋤繧帝幕蟋九＠縺ｾ縺吶・nCommand: {command_str}\n\n"

            yield from run_command(command_str)

        def start_training(
            project_path,
            model,
            dit,
            vae,
            te1,
            output_nm,
            training_mode,
            network_type,
            dim,
            lr,
            optimizer_type,
            optimizer_args,
            network_args,
            network_alpha,
            lr_warmup_steps,
            seed,
            max_grad_norm,
            epochs,
            save_n,
            flow_shift,
            swap,
            use_pinned_memory_for_block_swap,
            prec,
            grad_cp,
            fp8_s,
            fp8_l,
            full_bf16,
            fused_backward_pass,
            mem_eff_save,
            block_swap_optimizer_patch_params,
            add_args,
            should_sample_images,
            sample_every_n,
            sample_out_dir,
            sample_prompt_val,
            sample_negative_prompt_val,
            sample_w_val,
            sample_h_val,
            resume_from,
            lr_scheduler,
            lr_scheduler_args,
        ):
            import shlex

            model = _normalize_model_label(model)

            if training_mode not in [TRAINING_MODE_LORA, TRAINING_MODE_FINETUNE]:
                training_mode = TRAINING_MODE_LORA
            if not _is_zimage(model) and training_mode == TRAINING_MODE_FINETUNE:
                return "Error: Fine-tune mode is available only for Z-Image. / Fine-tune モードは Z-Image のみ対応です。"

            if not project_path:
                return "Error: Project directory not set."
            if not dit:
                return "Error: Base Model / DiT Path not set."
            if not os.path.exists(dit):
                return f"Error: Base Model / DiT file not found at {dit}"
            if not vae:
                return "Error: VAE Path not set (configure in Preprocessing)."
            if not te1:
                return "Error: Text Encoder 1 Path not set (configure in Preprocessing)."

            dataset_config = os.path.join(project_path, "dataset_config.toml")
            if not os.path.exists(dataset_config):
                return "Error: dataset_config.toml not found. Please generate it."

            output_dir = os.path.join(project_path, "models")
            logging_dir = os.path.join(project_path, "logs")

            if _is_zimage(model):
                arch_name = "zimage"
            elif model == "Qwen-Image":
                arch_name = "qwen_image"
            elif model.startswith("Flux.2"):
                arch_name = "flux_2"
            else:
                return f"Error: Unsupported model architecture: {model}"

            if training_mode == TRAINING_MODE_FINETUNE:
                script_path = os.path.join("src", "musubi_tuner", "zimage_train.py")
                network_module = None
            else:
                script_path = os.path.join("src", "musubi_tuner", f"{arch_name}_train_network.py")
                if network_type == "LoHa":
                    network_module = "networks.loha"
                elif network_type == "LoKr":
                    network_module = "networks.lokr"
                else:
                    network_type = "LoRA"
                    network_module = f"networks.lora_{arch_name}"

            save_project_settings(
                project_path,
                dit_path=dit,
                output_name=output_nm,
                training_mode=training_mode,
                network_type=network_type,
                network_dim=dim,
                learning_rate=lr,
                optimizer_type=optimizer_type,
                optimizer_args=optimizer_args,
                network_args=network_args,
                network_alpha=network_alpha,
                lr_warmup_steps=lr_warmup_steps,
                seed=seed,
                max_grad_norm=max_grad_norm,
                num_epochs=epochs,
                save_every_n_epochs=save_n,
                discrete_flow_shift=flow_shift,
                block_swap=swap,
                use_pinned_memory_for_block_swap=use_pinned_memory_for_block_swap,
                mixed_precision=prec,
                gradient_checkpointing=grad_cp,
                fp8_scaled=fp8_s,
                fp8_llm=fp8_l,
                full_bf16=full_bf16,
                fused_backward_pass=fused_backward_pass,
                mem_eff_save=mem_eff_save,
                block_swap_optimizer_patch_params=block_swap_optimizer_patch_params,
                vae_path=vae,
                text_encoder1_path=te1,
                additional_args=add_args,
                sample_images=should_sample_images,
                sample_every_n_epochs=sample_every_n,
                sample_output_dir=sample_out_dir,
                sample_prompt=sample_prompt_val,
                sample_negative_prompt=sample_negative_prompt_val,
                sample_w=sample_w_val,
                sample_h=sample_h_val,
                resume_from_checkpoint=resume_from,
                lr_scheduler=lr_scheduler,
                lr_scheduler_args=lr_scheduler_args,
            )

            inner_cmd = [
                "accelerate",
                "launch",
                "--num_cpu_threads_per_process",
                "1",
                "--mixed_precision",
                prec,
                "--dynamo_backend=no",
                "--gpu_ids",
                "all",
                "--machine_rank",
                "0",
                "--main_training_function",
                "main",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
                script_path,
            ]

            if training_mode == TRAINING_MODE_LORA:
                if model == "Flux.2 Dev":
                    inner_cmd.extend(["--model_version", "dev"])
                elif model == "Flux.2 Klein (4B)":
                    inner_cmd.extend(["--model_version", "klein-4b"])
                elif model == "Flux.2 Klein Base (4B)":
                    inner_cmd.extend(["--model_version", "klein-base-4b"])

            timestep_sampling = "flux2_shift" if model.startswith("Flux.2") else "shift"

            inner_cmd.extend(
                [
                    "--dit",
                    dit,
                    "--vae",
                    vae,
                    "--text_encoder",
                    te1,
                    "--dataset_config",
                    dataset_config,
                    "--output_dir",
                    output_dir,
                    "--output_name",
                    output_nm,
                    "--optimizer_type",
                    optimizer_type,
                    "--learning_rate",
                    str(lr),
                    "--lr_warmup_steps",
                    str(lr_warmup_steps),
                    "--seed",
                    str(int(seed)),
                    "--max_grad_norm",
                    str(max_grad_norm),
                    "--lr_scheduler",
                    lr_scheduler,
                    "--max_train_epochs",
                    str(int(epochs)),
                    "--save_every_n_epochs",
                    str(int(save_n)),
                    "--timestep_sampling",
                    timestep_sampling,
                    "--weighting_scheme",
                    "none",
                    "--discrete_flow_shift",
                    str(flow_shift),
                    "--max_data_loader_n_workers",
                    "2",
                    "--persistent_data_loader_workers",
                    "--logging_dir",
                    logging_dir,
                    "--log_with",
                    "tensorboard",
                ]
            )

            if training_mode == TRAINING_MODE_LORA:
                inner_cmd.extend(
                    [
                        "--network_module",
                        network_module,
                        "--network_dim",
                        str(int(dim)),
                        "--network_alpha",
                        str(network_alpha),
                    ]
                )

            if optimizer_args and optimizer_args.strip():
                try:
                    inner_cmd.append("--optimizer_args")
                    inner_cmd.extend(shlex.split(optimizer_args))
                except Exception as e:
                    return f"Error parsing optimizer args: {str(e)}"

            if lr_scheduler_args and lr_scheduler_args.strip():
                try:
                    inner_cmd.append("--lr_scheduler_args")
                    inner_cmd.extend(shlex.split(lr_scheduler_args))
                except Exception as e:
                    return f"Error parsing scheduler args: {str(e)}"

            if training_mode == TRAINING_MODE_LORA and network_args and network_args.strip():
                try:
                    inner_cmd.append("--network_args")
                    inner_cmd.extend(shlex.split(network_args))
                except Exception as e:
                    return f"Error parsing network args: {str(e)}"

            if should_sample_images:
                sample_prompt_path = os.path.join(project_path, "sample_prompt.txt")
                templates = {
                    "Qwen-Image": "{prompt} --n {neg} --w {w} --h {h} --fs 2.2 --s 20 --l 4.0 --d 1234",
                    "Z-Image": "{prompt} --n {neg} --w {w} --h {h} --fs 3.0 --s 20 --l 5.0 --d 1234",
                    "Flux.2 Dev": "{prompt} --n {neg} --w {w} --h {h} --s 50 --g 4.0 --d 1234",
                    "Flux.2 Klein (4B)": "{prompt} --n {neg} --w {w} --h {h} --s 4 --g 1.0 --d 1234",
                    "Flux.2 Klein Base (4B)": "{prompt} --n {neg} --w {w} --h {h} --s 50 --g 4.0 --d 1234",
                }
                template = templates.get(model, templates["Z-Image"])
                prompt_str = (sample_prompt_val or "").replace("\n", " ").strip()
                neg_str = (sample_negative_prompt_val or "").replace("\n", " ").strip()
                try:
                    w_int = int(sample_w_val)
                    h_int = int(sample_h_val)
                except Exception:
                    return "Error: Sample width/height must be integers."

                line = template.format(prompt=prompt_str, neg=neg_str, w=w_int, h=h_int)
                try:
                    with open(sample_prompt_path, "w", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception as e:
                    return f"Error writing sample_prompt.txt: {str(e)}"

                inner_cmd.extend(
                    [
                        "--sample_prompts",
                        sample_prompt_path,
                        "--sample_at_first",
                        "--sample_every_n_epochs",
                        str(int(sample_every_n)),
                    ]
                )
                if sample_out_dir and sample_out_dir.strip():
                    inner_cmd.extend(["--sample_output_dir", sample_out_dir.strip()])

            if prec != "no":
                inner_cmd.extend(["--mixed_precision", prec])

            if grad_cp:
                inner_cmd.append("--gradient_checkpointing")

            if fp8_s:
                inner_cmd.append("--fp8_base")
                inner_cmd.append("--fp8_scaled")

            if fp8_l:
                if _is_zimage(model):
                    inner_cmd.append("--fp8_llm")
                elif model == "Qwen-Image":
                    inner_cmd.append("--fp8_vl")
                elif model.startswith("Flux.2") and model != "Flux.2 Dev":
                    inner_cmd.append("--fp8_text_encoder")

            if swap > 0:
                inner_cmd.extend(["--blocks_to_swap", str(int(swap))])
                if use_pinned_memory_for_block_swap:
                    inner_cmd.append("--use_pinned_memory_for_block_swap")

            inner_cmd.append("--sdpa")
            inner_cmd.append("--split_attn")

            if training_mode == TRAINING_MODE_FINETUNE:
                if full_bf16:
                    inner_cmd.append("--full_bf16")
                if fused_backward_pass:
                    inner_cmd.append("--fused_backward_pass")
                if mem_eff_save:
                    inner_cmd.append("--mem_eff_save")
                if block_swap_optimizer_patch_params:
                    inner_cmd.append("--block_swap_optimizer_patch_params")

            if resume_from and resume_from.strip():
                inner_cmd.extend(["--resume", resume_from.strip()])

            if add_args:
                try:
                    split_args = shlex.split(add_args)
                    inner_cmd.extend(split_args)
                except Exception as e:
                    return f"Error parsing additional arguments: {str(e)}"

            inner_cmd_str = subprocess.list2cmdline(inner_cmd)
            final_cmd_str = f"{inner_cmd_str} & echo. & echo Training finished. Press any key to close this window... & pause >nul"

            try:
                flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
                subprocess.Popen(["cmd", "/c", final_cmd_str], creationflags=flags, shell=False)
                return f"Training started in a new window!\nCommand: {inner_cmd_str}"
            except Exception as e:
                return f"Error starting training: {str(e)}"

        def update_model_info(model):
            if _is_zimage(model):
                return i18n("desc_training_zimage")
            elif model == "Qwen-Image":
                return i18n("desc_qwen_notes")
            elif model.startswith("Flux.2"):
                return i18n("desc_training_flux2")
            return ""

        def _mode_visibility_updates(mode):
            is_finetune = mode == TRAINING_MODE_FINETUNE
            return [
                gr.update(visible=not is_finetune),  # network_type
                gr.update(visible=not is_finetune),  # network_dim
                gr.update(visible=not is_finetune),  # network_alpha
                gr.update(visible=not is_finetune),  # network_args
                gr.update(visible=is_finetune),  # fine-tune options row
            ]

        def update_training_mode_visibility(mode):
            if mode not in [TRAINING_MODE_LORA, TRAINING_MODE_FINETUNE]:
                mode = TRAINING_MODE_LORA
            return _mode_visibility_updates(mode)

        def update_training_mode_options(model, current_mode):
            model = _normalize_model_label(model)
            if _is_zimage(model):
                choices = [TRAINING_MODE_LORA, TRAINING_MODE_FINETUNE]
            else:
                choices = [TRAINING_MODE_LORA]

            selected_mode = current_mode if current_mode in choices else TRAINING_MODE_LORA
            return [gr.update(choices=choices, value=selected_mode), *_mode_visibility_updates(selected_mode)]

        for name, component in [
            ("preset_name", preset_name),
            ("load_preset_dd", load_preset_dd),
            ("preset_apply_paths", preset_apply_paths),
            ("preset_status", preset_status),
            ("project_dir", project_dir),
            ("project_status", project_status),
            ("model_arch", model_arch),
            ("vram_size", vram_size),
            ("comfy_models_dir", comfy_models_dir),
            ("models_status", models_status),
            ("quick_status", quick_status),
            ("resolution_w", resolution_w),
            ("resolution_h", resolution_h),
            ("batch_size", batch_size),
            ("control_directory", control_directory),
            ("control_res_w", control_res_w),
            ("control_res_h", control_res_h),
            ("no_resize_control", no_resize_control),
            ("image_directory", image_directory),
            ("cache_directory", cache_directory),
            ("caption_extension", caption_extension),
            ("num_repeats", num_repeats),
            ("enable_bucket", enable_bucket),
            ("bucket_no_upscale", bucket_no_upscale),
            ("dataset_status", dataset_status),
            ("toml_preview", toml_preview),
            ("auto_detect_status", auto_detect_status),
            ("vae_path", vae_path),
            ("text_encoder1_path", text_encoder1_path),
            ("text_encoder2_path", text_encoder2_path),
            ("caching_output", caching_output),
            ("training_model_info", training_model_info),
            ("dit_path", dit_path),
            ("output_name", output_name),
            ("training_mode", training_mode),
            ("network_type", network_type),
            ("network_dim", network_dim),
            ("learning_rate", learning_rate),
            ("optimizer_type", optimizer_type),
            ("optimizer_args", optimizer_args),
            ("network_args", network_args),
            ("lr_scheduler", lr_scheduler),
            ("lr_scheduler_args", lr_scheduler_args),
            ("network_alpha", network_alpha),
            ("lr_warmup_steps", lr_warmup_steps),
            ("seed", seed),
            ("max_grad_norm", max_grad_norm),
            ("num_epochs", num_epochs),
            ("save_every_n_epochs", save_every_n_epochs),
            ("discrete_flow_shift", discrete_flow_shift),
            ("block_swap", block_swap),
            ("use_pinned_memory_for_block_swap", use_pinned_memory_for_block_swap),
            ("mixed_precision", mixed_precision),
            ("gradient_checkpointing", gradient_checkpointing),
            ("fp8_scaled", fp8_scaled),
            ("fp8_llm", fp8_llm),
            ("full_bf16", full_bf16),
            ("fused_backward_pass", fused_backward_pass),
            ("mem_eff_save", mem_eff_save),
            ("block_swap_optimizer_patch_params", block_swap_optimizer_patch_params),
            ("vram_estimate", vram_estimate),
            ("sample_images", sample_images),
            ("sample_prompt", sample_prompt),
            ("sample_negative_prompt", sample_negative_prompt),
            ("sample_w", sample_w),
            ("sample_h", sample_h),
            ("sample_every_n", sample_every_n),
            ("sample_output_dir", sample_output_dir),
            ("additional_args", additional_args),
            ("resume_path", resume_path),
            ("training_status", training_status),
            ("input_lora", input_lora),
            ("output_comfy_lora", output_comfy_lora),
            ("lokr_rank", lokr_rank),
            ("conversion_log", conversion_log),
        ]:
            register(name, component)

        # Event wiring moved to end to prevent UnboundLocalError
        init_evt = init_btn.click(
            fn=init_project,
            inputs=[project_dir],
            outputs=[project_status, *components_for(INIT_OUTPUT_KEYS)],
        )
        init_evt.then(
            fn=update_training_mode_options,
            inputs=[model_arch, training_mode],
            outputs=[training_mode, network_type, network_dim, network_alpha, network_args, finetune_opts_row],
        )

        model_arch.change(fn=update_model_info, inputs=[model_arch], outputs=[training_model_info])
        model_arch.change(
            fn=update_training_mode_options,
            inputs=[model_arch, training_mode],
            outputs=[training_mode, network_type, network_dim, network_alpha, network_args, finetune_opts_row],
        )
        training_mode.change(
            fn=update_training_mode_visibility,
            inputs=[training_mode],
            outputs=[network_type, network_dim, network_alpha, network_args, finetune_opts_row],
        )

        gen_toml_btn.click(
            fn=generate_config,
            inputs=[
                project_dir,
                resolution_w,
                resolution_h,
                batch_size,
                model_arch,
                vram_size,
                comfy_models_dir,
                vae_path,
                text_encoder1_path,
                text_encoder2_path,
                control_directory,
                control_res_w,
                control_res_h,
                no_resize_control,
                image_directory,
                cache_directory,
                caption_extension,
                num_repeats,
                enable_bucket,
                bucket_no_upscale,
            ],
            outputs=[dataset_status, toml_preview],
        )

        validate_models_btn.click(fn=validate_models_dir, inputs=[comfy_models_dir], outputs=[models_status])

        quick_setup_evt = quick_setup_btn.click(
            fn=quick_setup,
            inputs=[project_dir, model_arch, vram_size, comfy_models_dir],
            outputs=components_for(QUICK_SETUP_OUTPUT_KEYS),
        )
        quick_setup_evt.then(
            fn=update_training_mode_options,
            inputs=[model_arch, training_mode],
            outputs=[training_mode, network_type, network_dim, network_alpha, network_args, finetune_opts_row],
        )

        check_missing_btn.click(
            fn=check_missing,
            inputs=[project_dir, comfy_models_dir, vae_path, text_encoder1_path, dit_path],
            outputs=[quick_status],
        )

        browse_project_dir.click(fn=browse_dir, inputs=[project_dir], outputs=[project_dir])
        browse_comfy_dir.click(fn=browse_dir, inputs=[comfy_models_dir], outputs=[comfy_models_dir])
        browse_control_dir.click(fn=browse_dir, inputs=[control_directory], outputs=[control_directory])
        browse_image_dir.click(fn=browse_dir, inputs=[image_directory], outputs=[image_directory])
        browse_cache_dir.click(fn=browse_dir, inputs=[cache_directory], outputs=[cache_directory])

        open_presets_btn.click(fn=open_presets_folder, outputs=[preset_status])
        open_project_btn.click(fn=open_project_folder, inputs=[project_dir], outputs=[project_status])
        open_training_btn.click(fn=open_training_folder, inputs=[project_dir], outputs=[project_status])
        open_logs_btn.click(fn=open_logs_folder, inputs=[project_dir], outputs=[project_status])

        browse_vae.click(fn=browse_file, inputs=[vae_path], outputs=[vae_path])
        browse_te1.click(fn=browse_file, inputs=[text_encoder1_path], outputs=[text_encoder1_path])
        browse_te2.click(fn=browse_file, inputs=[text_encoder2_path], outputs=[text_encoder2_path])
        browse_dit.click(fn=browse_file, inputs=[dit_path], outputs=[dit_path])

        browse_sample_out.click(fn=browse_dir, inputs=[sample_output_dir], outputs=[sample_output_dir])
        browse_resume.click(fn=browse_dir, inputs=[resume_path], outputs=[resume_path])
        browse_input_lora.click(fn=browse_file, inputs=[input_lora], outputs=[input_lora])
        browse_output_lora.click(fn=browse_save, inputs=[output_comfy_lora], outputs=[output_comfy_lora])

        set_rec_settings_btn.click(
            fn=set_recommended_settings,
            inputs=[project_dir, model_arch, vram_size],
            outputs=components_for(RECOMMENDED_KEYS),
        )

        set_preprocessing_defaults_btn.click(
            fn=set_preprocessing_defaults,
            inputs=[project_dir, comfy_models_dir, model_arch],
            outputs=[vae_path, text_encoder1_path, text_encoder2_path],
        )

        auto_detect_paths_btn.click(
            fn=auto_detect_paths,
            inputs=[project_dir, comfy_models_dir, model_arch],
            outputs=[vae_path, text_encoder1_path, text_encoder2_path, dit_path, auto_detect_status],
        )

        set_post_proc_defaults_btn.click(
            fn=set_post_processing_defaults, inputs=[project_dir, output_name], outputs=[input_lora, output_comfy_lora]
        )

        set_training_defaults_evt = set_training_defaults_btn.click(
            fn=set_training_defaults,
            inputs=[project_dir, comfy_models_dir, model_arch, vram_size],
            outputs=components_for(TRAINING_DEFAULT_KEYS),
        )
        set_training_defaults_evt.then(
            fn=update_training_mode_options,
            inputs=[model_arch, training_mode],
            outputs=[training_mode, network_type, network_dim, network_alpha, network_args, finetune_opts_row],
        )

        est_vram_btn.click(
            fn=estimate_vram,
            inputs=[
                model_arch,
                resolution_w,
                resolution_h,
                batch_size,
                mixed_precision,
                gradient_checkpointing,
                fp8_scaled,
                fp8_llm,
                block_swap,
            ],
            outputs=[vram_estimate],
        )

        cache_latents_btn.click(
            fn=cache_latents,
            inputs=[
                project_dir,
                vae_path,
                text_encoder1_path,
                text_encoder2_path,
                model_arch,
                comfy_models_dir,
                resolution_w,
                resolution_h,
                batch_size,
                vram_size,
            ],
            outputs=[caching_output],
        )

        cache_text_btn.click(
            fn=cache_text_encoder,
            inputs=[
                project_dir,
                text_encoder1_path,
                text_encoder2_path,
                vae_path,
                model_arch,
                comfy_models_dir,
                resolution_w,
                resolution_h,
                batch_size,
                vram_size,
                fp8_llm,
            ],
            outputs=[caching_output],
        )

        start_training_btn.click(
            fn=start_training,
            inputs=[
                project_dir,
                model_arch,
                dit_path,
                vae_path,
                text_encoder1_path,
                output_name,
                training_mode,
                network_type,
                network_dim,
                learning_rate,
                optimizer_type,
                optimizer_args,
                network_args,
                network_alpha,
                lr_warmup_steps,
                seed,
                max_grad_norm,
                num_epochs,
                save_every_n_epochs,
                discrete_flow_shift,
                block_swap,
                use_pinned_memory_for_block_swap,
                mixed_precision,
                gradient_checkpointing,
                fp8_scaled,
                fp8_llm,
                full_bf16,
                fused_backward_pass,
                mem_eff_save,
                block_swap_optimizer_patch_params,
                additional_args,
                sample_images,
                sample_every_n,
                sample_output_dir,
                sample_prompt,
                sample_negative_prompt,
                sample_w,
                sample_h,
                resume_path, # Added
                lr_scheduler, # Added
                lr_scheduler_args, # Added
            ],
            outputs=[training_status],
        )
        
        # New Feature Event Wiring
        
        save_preset_btn.click(
            fn=save_preset,
            inputs=[preset_name] + components_for(PRESET_COMPONENT_KEYS),
            outputs=[preset_status]
        )
        
        # Chain save with refresh list
        save_preset_btn.click(fn=refresh_preset_dropdown, outputs=[load_preset_dd])

        load_preset_evt = load_preset_btn.click(
             fn=load_preset,
             inputs=[load_preset_dd, preset_apply_paths],
             outputs=[preset_status, *components_for(PRESET_OUTPUT_COMPONENT_KEYS)]
        )
        load_preset_evt.then(
            fn=update_training_mode_options,
            inputs=[model_arch, training_mode],
            outputs=[training_mode, network_type, network_dim, network_alpha, network_args, finetune_opts_row],
        )
        
        refresh_preset_btn.click(fn=refresh_preset_dropdown, outputs=[load_preset_dd])
        
        tensorboard_btn.click(fn=launch_tensorboard, inputs=[project_dir], outputs=[training_status])


        convert_btn.click(
            fn=convert_lora_to_comfy,
            inputs=[
                project_dir,
                input_lora,
                output_comfy_lora,
                model_arch,
                comfy_models_dir,
                resolution_w,
                resolution_h,
                batch_size,
                vae_path,
                text_encoder1_path,
                text_encoder2_path,
                lokr_rank,
            ],
            outputs=[conversion_log],
        )

    return demo


if __name__ == "__main__":
    demo = construct_ui()
    launch_kwargs = {}
    if LAUNCH_SUPPORTS_I18N:
        launch_kwargs["i18n"] = i18n
    if LAUNCH_SUPPORTS_THEME:
        launch_kwargs["theme"] = gr.themes.Soft()
    if LAUNCH_SUPPORTS_CSS:
        launch_kwargs["css"] = APP_CSS
    demo.launch(**launch_kwargs)

