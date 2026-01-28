import glob
import gradio as gr
import inspect
import os
import warnings
import toml
import subprocess
import sys
import json
import shlex
from config_manager import ConfigManager
from i18n_data import I18N_DATA

config_manager = ConfigManager()


i18n = gr.I18n(en=I18N_DATA["en"], ja=I18N_DATA["ja"])

try:
    _launch_params = inspect.signature(gr.Blocks.launch).parameters
except (ValueError, TypeError):
    _launch_params = {}

LAUNCH_SUPPORTS_THEME = "theme" in _launch_params
LAUNCH_SUPPORTS_CSS = "css" in _launch_params

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
@import url("https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap");

:root {
  --bg-1: #f6f2ea;
  --bg-2: #e9f2ff;
  --card: #ffffff;
  --ink: #1f2937;
  --muted: #6b7280;
  --accent: #0f766e;
  --border: #e5e7eb;
  --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
}

.gradio-container {
  font-family: "Sora", "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Meiryo", sans-serif;
  background: radial-gradient(1200px 600px at 90% -20%, var(--bg-2), transparent),
              radial-gradient(1000px 500px at -10% 0%, #fbeee3, transparent),
              linear-gradient(180deg, #f9fafb 0%, #f3f4f6 100%);
  color: var(--ink);
}

#app-header h1 {
  letter-spacing: -0.02em;
  font-weight: 700;
}

#app-desc {
  color: var(--muted);
  margin-bottom: 12px;
}

.section-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px;
  box-shadow: var(--shadow);
}

.gr-button {
  border-radius: 10px;
}

.gr-button.primary {
  background: var(--accent);
  border-color: var(--accent);
}

.path-row {
  align-items: flex-end;
}

.subtle-note {
  color: var(--muted);
  font-size: 0.9rem;
}
"""

def construct_ui():
    # --- Preset Management ---
    PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")
    os.makedirs(PRESETS_DIR, exist_ok=True)

    def ensure_default_presets():
        zimage_vram = "24"
        zimage_resolution = config_manager.get_resolution("Z-Image-Turbo")
        zimage_batch = config_manager.get_batch_size("Z-Image-Turbo", zimage_vram)
        zimage_defaults = config_manager.get_training_defaults("Z-Image-Turbo", zimage_vram, "")

        preset = {
            "project_dir": "",
            "model_arch": "Z-Image-Turbo",
            "vram_size": zimage_vram,
            "comfy_models_dir": "",
            "resolution_w": zimage_resolution[0],
            "resolution_h": zimage_resolution[1],
            "batch_size": zimage_batch,
            "control_directory": "",
            "control_resolution_w": 0,
            "control_resolution_h": 0,
            "no_resize_control": False,
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
        }

        defaults = {"Z-Image (default)": preset}
        for name, data in defaults.items():
            path = os.path.join(PRESETS_DIR, f"{name}.json")
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

    ensure_default_presets()
    initial_fp8_llm = config_manager.get_training_defaults("Flux.2 Klein (4B)", "24", "").get("fp8_llm", False)

    def get_preset_list():
        return [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(PRESETS_DIR, "*.json"))]

    def save_preset(name, *args):
        if not name:
            return i18n("msg_preset_error").format(e="Name is empty")
        try:
            # args order corresponds to the inputs list of save button
            keys = [
                "project_dir", "model_arch", "vram_size", "comfy_models_dir",
                "resolution_w", "resolution_h", "batch_size",
                "control_directory", "control_resolution_w", "control_resolution_h", "no_resize_control",
                "vae_path", "text_encoder1_path", "text_encoder2_path",
                "dit_path", "output_name", "dim", "lr", "optimizer_type", "optimizer_args", "lr_scheduler", "lr_scheduler_args",
                "network_alpha", "lr_warmup_steps", "seed", "max_grad_norm",
                "epochs", "save_every",
                "flow_shift", "block_swap", "pinned", "mixed_precision", "grad_checkpointing",
                "fp8_scaled", "fp8_llm", "additional_args",
                "sample_images", "sample_every", "sample_output_dir", "sample_prompt", "sample_neg", "sample_w", "sample_h",
                "input_lora", "output_comfy"
            ]
            data = dict(zip(keys, args))
            path = os.path.join(PRESETS_DIR, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return i18n("msg_preset_saved").format(name=name)
        except Exception as e:
            return i18n("msg_preset_error").format(e=str(e))

    def load_preset(name):
        if not name:
            return [gr.update()] * 45 # Return no updates
        try:
            path = os.path.join(PRESETS_DIR, f"{name}.json")
            if not os.path.exists(path):
                return [gr.update()] * 45
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Return values in order. Use get with defaults.
            return [
                gr.update(), data.get("model_arch", "Flux.2 Klein (4B)"), data.get("vram_size", "24"), data.get("comfy_models_dir", ""),
                data.get("resolution_w", 1024), data.get("resolution_h", 1024), data.get("batch_size", 1),
                data.get("control_directory", ""), data.get("control_resolution_w", 0), data.get("control_resolution_h", 0), data.get("no_resize_control", False),
                data.get("vae_path", ""), data.get("text_encoder1_path", ""), data.get("text_encoder2_path", ""),
                data.get("dit_path", ""), data.get("output_name", "my_lora"), data.get("dim", 32), data.get("lr", 1e-4),
                data.get("optimizer_type", "adamw8bit"), data.get("optimizer_args", ""),
                data.get("lr_scheduler", "constant"), data.get("lr_scheduler_args", ""),
                data.get("network_alpha", 1), data.get("lr_warmup_steps", 0), data.get("seed", 42), data.get("max_grad_norm", 1.0),
                data.get("epochs", 16), data.get("save_every", 1),
                data.get("flow_shift", 1.0), data.get("block_swap", 0), data.get("pinned", False), data.get("mixed_precision", "bf16"), data.get("grad_checkpointing", True),
                data.get("fp8_scaled", False), data.get("fp8_llm", False), data.get("additional_args", ""),
                data.get("sample_images", True), data.get("sample_every", 1), data.get("sample_output_dir", ""), data.get("sample_prompt", ""), data.get("sample_neg", ""), data.get("sample_w", 1024), data.get("sample_h", 1024),
                data.get("input_lora", ""), data.get("output_comfy", "")
            ]
        except Exception as e:
             # In case of error, just don't update anything or handle gracefully
             print(f"Error loading preset: {e}")
             return [gr.update()] * 45

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

        # Presets Section
        with gr.Accordion(i18n("header_presets"), open=False, elem_classes=["section-card"]):
            with gr.Row():
                preset_name = gr.Textbox(label=i18n("lbl_preset_name"), scale=2)
                save_preset_btn = gr.Button(i18n("btn_save_preset"), scale=1)
            with gr.Row():
                load_preset_dd = gr.Dropdown(label=i18n("lbl_load_preset"), choices=get_preset_list(), scale=2)
                load_preset_btn = gr.Button(i18n("btn_load_preset"), scale=1)
                refresh_preset_btn = gr.Button(i18n("btn_refresh_presets"), scale=0)
            preset_status = gr.Markdown("")

        with gr.Accordion(i18n("acc_project"), open=True, elem_classes=["section-card"]):
            gr.Markdown(i18n("desc_project"))
            with gr.Row(elem_classes=["path-row"]):
                project_dir = gr.Textbox(label=i18n("lbl_proj_dir"), placeholder=i18n("ph_proj_dir"), max_lines=1, scale=8)
                browse_project_dir = gr.Button(i18n("btn_browse"), scale=1)

            # Placeholder for project initialization or loading
            init_btn = gr.Button(i18n("btn_init_project"))
            project_status = gr.Markdown("")

        with gr.Accordion(i18n("acc_model"), open=False, elem_classes=["section-card"]):
            gr.Markdown(i18n("desc_model"))
            with gr.Row():
                model_arch = gr.Dropdown(
                    label=i18n("lbl_model_arch"),
                    choices=[
                        "Flux.2 Klein (4B)",
                        "Flux.2 Klein Base (4B)",
                        "Flux.2 Dev",
                        "Qwen-Image",
                        "Z-Image-Turbo",
                    ],
                    value="Flux.2 Klein (4B)",
                )
                vram_size = gr.Dropdown(label=i18n("lbl_vram"), choices=["12", "16", "24", "32", ">32"], value="24")

            with gr.Row(elem_classes=["path-row"]):
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
            with gr.Row(elem_classes=["path-row"]):
                control_directory = gr.Textbox(label=i18n("lbl_control_dir"), placeholder=i18n("ph_control_dir"), max_lines=1, scale=8)
                browse_control_dir = gr.Button(i18n("btn_browse"), scale=1)
            with gr.Row():
                control_res_w = gr.Number(label=i18n("lbl_control_res_w"), value=0, precision=0)
                control_res_h = gr.Number(label=i18n("lbl_control_res_h"), value=0, precision=0)
                no_resize_control = gr.Checkbox(label=i18n("lbl_no_resize_control"), value=False)

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
                    return (
                        "Please enter a project directory path.",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                try:
                    os.makedirs(os.path.join(path, "training"), exist_ok=True)

                    # Load settings if available
                    settings = load_project_settings(path)
                    new_model = settings.get("model_arch", "Flux.2 Klein (4B)")
                    new_vram = settings.get("vram_size", "16")
                    new_comfy = settings.get("comfy_models_dir", "")
                    new_w = settings.get("resolution_w", 1024)
                    new_h = settings.get("resolution_h", 1024)
                    new_batch = settings.get("batch_size", 1)
                    new_control_dir = settings.get("control_directory", "")
                    new_control_w = settings.get("control_resolution_w", 0)
                    new_control_h = settings.get("control_resolution_h", 0)
                    new_no_resize_control = settings.get("no_resize_control", False)
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
                    new_epochs = settings.get("num_epochs", 16)
                    new_save_n = settings.get("save_every_n_epochs", 1)
                    new_flow = settings.get("discrete_flow_shift", 2.0)
                    new_swap = settings.get("block_swap", 0)
                    new_use_pinned_memory_for_block_swap = settings.get("use_pinned_memory_for_block_swap", False)
                    new_prec = settings.get("mixed_precision", "bf16")
                    new_grad_cp = settings.get("gradient_checkpointing", True)
                    new_fp8_s = settings.get("fp8_scaled", True)
                    new_fp8_l = settings.get("fp8_llm", False)
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

                    # Load dataset config content
                    preview_content = load_dataset_config_content(path)

                    msg = f"Project initialized at {path}. "
                    if settings:
                        msg += " Settings loaded."
                    msg += " 'training' folder ready. Configure the dataset in the 'training' folder. Images and caption files (same name as image, extension is '.txt') should be placed in the 'training' folder."
                    msg += "\n\nプロジェクトが初期化されました。"
                    if settings:
                        msg += "設定が読み込まれました。"
                    msg += "'training' フォルダが準備されました。画像とキャプションファイル（画像と同じファイル名で拡張子は '.txt'）を配置してください。"

                    return (
                        msg,
                        new_model,
                        new_vram,
                        new_comfy,
                        new_w,
                        new_h,
                        new_batch,
                        new_control_dir,
                        new_control_w,
                        new_control_h,
                        new_no_resize_control,
                        preview_content,
                        new_vae,
                        new_te1,
                        new_te2,
                        new_dit,
                        new_out_nm,
                        new_dim,
                        new_lr,
                        new_optimizer_type,
                        new_optimizer_args,
                        new_lr_scheduler,
                        new_lr_scheduler_args,
                        new_network_alpha,
                        new_lr_warmup_steps,
                        new_seed,
                        new_max_grad_norm,
                        new_epochs,
                        new_save_n,
                        new_flow,
                        new_swap,
                        new_use_pinned_memory_for_block_swap,
                        new_prec,
                        new_grad_cp,
                        new_fp8_s,
                        new_fp8_l,
                        new_add_args,
                        new_sample_enable,
                        new_sample_every_n,
                        new_sample_output_dir,
                        new_sample_prompt,
                        new_sample_negative,
                        new_sample_w,
                        new_sample_h,
                        new_in_lora,
                        new_out_comfy,
                    )
                except Exception as e:
                    return (
                        f"Error initializing project: {str(e)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

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
            ):
                if not project_path:
                    return "Error: Project directory not specified.\nエラー: プロジェクトディレクトリが指定されていません。", ""

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
                )

                # Normalize paths
                project_path = os.path.abspath(project_path)
                image_dir = os.path.join(project_path, "training").replace("\\", "/")
                cache_dir = os.path.join(project_path, "cache").replace("\\", "/")

                toml_content = f"""# Auto-generated by Musubi Tuner GUI

[general]
resolution = [{int(w)}, {int(h)}]
caption_extension = ".txt"
batch_size = {int(batch)}
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{image_dir}"
cache_directory = "{cache_dir}"
num_repeats = 1
"""
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
                    return f"Successfully generated config at / 設定ファイルが作成されました: {config_path}", toml_content
                except Exception as e:
                    return f"Error generating config / 設定ファイルの生成に失敗しました: {str(e)}", ""

        with gr.Accordion(i18n("acc_preprocessing"), open=False, elem_classes=["section-card"]):
            gr.Markdown(i18n("desc_preprocessing"))
            with gr.Row():
                set_preprocessing_defaults_btn = gr.Button(i18n("btn_set_paths"))
                auto_detect_paths_btn = gr.Button(i18n("btn_auto_detect_paths"))
            auto_detect_status = gr.Markdown("")
            with gr.Row(elem_classes=["path-row"]):
                vae_path = gr.Textbox(label=i18n("lbl_vae_path"), placeholder=i18n("ph_vae_path"), max_lines=1, scale=8)
                browse_vae = gr.Button(i18n("btn_browse"), scale=1)
            with gr.Row(elem_classes=["path-row"]):
                text_encoder1_path = gr.Textbox(label=i18n("lbl_te1_path"), placeholder=i18n("ph_te1_path"), max_lines=1, scale=8)
                browse_te1 = gr.Button(i18n("btn_browse"), scale=1)
            with gr.Row(elem_classes=["path-row"]):
                text_encoder2_path = gr.Textbox(label=i18n("lbl_te2_path"), placeholder=i18n("ph_te2_path"), max_lines=1, scale=8)
                browse_te2 = gr.Button(i18n("btn_browse"), scale=1)

            with gr.Row():
                cache_latents_btn = gr.Button(i18n("btn_cache_latents"))
                cache_text_btn = gr.Button(i18n("btn_cache_text"))

            # Simple output area for caching logs
            caching_output = gr.Textbox(label=i18n("lbl_cache_log"), lines=10, interactive=False)

            def validate_models_dir(path):
                if not path:
                    return "Please enter a ComfyUI models directory. / ComfyUIのmodelsディレクトリを入力してください。"

                required_subdirs = ["diffusion_models", "vae", "text_encoders"]
                missing = []
                for d in required_subdirs:
                    if not os.path.exists(os.path.join(path, d)):
                        missing.append(d)

                if missing:
                    return f"Error: Missing subdirectories in models folder / modelsフォルダに以下のサブディレクトリが見つかりません: {', '.join(missing)}"

                return "Valid ComfyUI models directory structure found / 有効なComfyUI modelsディレクトリ構造が見つかりました。"

            def set_recommended_settings(project_path, model_arch, vram_val):
                w, h = config_manager.get_resolution(model_arch)
                recommended_batch_size = config_manager.get_batch_size(model_arch, vram_val)

                if project_path:
                    save_project_settings(project_path, resolution_w=w, resolution_h=h, batch_size=recommended_batch_size)
                return w, h, recommended_batch_size

            def set_preprocessing_defaults(project_path, comfy_models_dir, model_arch):
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
                        matches = glob.glob(os.path.join(search_dir, pattern))
                        matches.sort()
                        if matches:
                            return matches[0]
                    return ""

                if model_arch == "Flux.2 Dev":
                    dit_patterns = ["flux2-dev.safetensors", "flux2_dev.safetensors", "*flux2*dev*.safetensors"]
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
                        "*flux2*klein*4b*.safetensors",
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
                        "*flux2*klein*base*4b*.safetensors",
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
                else:  # Z-Image-Turbo (default)
                    dit_patterns = ["z_image_de_turbo_v1_bf16.safetensors", "*z*image*de*turbo*bf16*.safetensors"]
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

                sample_w_default, sample_h_default = config_manager.get_resolution(model_arch)
                sample_out = "" # Default empty

                if project_path:
                    save_project_settings(
                        project_path,
                        dit_path=dit_default,
                        network_dim=dim,
                        learning_rate=lr,
                        optimizer_type=optimizer_type,
                        optimizer_args=optimizer_args,
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
                        vram_size=vram_val,  # Ensure VRAM size is saved
                        sample_every_n_epochs=sample_every_n_epochs,
                        sample_w=sample_w_default,
                        sample_h=sample_h_default,
                        sample_output_dir=sample_out,
                    )

                return (
                    dit_default,
                    dim,
                    lr,
                    optimizer_type,
                    optimizer_args,
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
                    sample_every_n_epochs,
                    sample_w_default,
                    sample_h_default,
                    sample_out,
                )

            def estimate_vram(
                model_arch,
                w,
                h,
                batch,
                prec,
                grad_cp,
                fp8_s,
                fp8_l,
                swap,
            ):
                try:
                    w = int(w)
                    h = int(h)
                    batch = int(batch)
                except Exception:
                    return i18n("msg_est_vram_note")

                if w <= 0 or h <= 0 or batch <= 0:
                    return i18n("msg_est_vram_note")

                base_map = {
                    "Z-Image-Turbo": 8.0,
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
                        "Z-Image-Turbo": 28,
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
                    f"**{i18n('msg_est_vram_title')}**: ~{low:.1f}–{high:.1f} GB\n\n"
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

                return (
                    w,
                    h,
                    batch,
                    _prefer_auto(auto_vae, vae_default),
                    _prefer_auto(auto_te1, te1_default),
                    _prefer_auto(auto_te2, te2_default),
                    _prefer_auto(auto_dit, dit_default),
                    dim,
                    lr,
                    optimizer_type,
                    optimizer_args,
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
                    sample_every_n_epochs,
                    sample_w_default,
                    sample_h_default,
                    sample_out,
                    update_model_info(model_arch),
                    status_msg,
                )

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
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        shell=True,
                        text=True,
                        encoding="utf-8",
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )

                    output_log = command + "\n\n"
                    for line in process.stdout:
                        output_log += line
                        yield output_log

                    process.wait()
                    if process.returncode != 0:
                        output_log += (
                            f"\nError: Process exited with code / プロセスが次のコードでエラー終了しました: {process.returncode}"
                        )
                        yield output_log
                    else:
                        output_log += "\nProcess completed successfully / プロセスが正常に完了しました"
                        yield output_log

                except Exception as e:
                    yield f"Error executing command / コマンドの実行中にエラーが発生しました: {str(e)}"

            def cache_latents(project_path, vae_path_val, te1, te2, model, comfy, w, h, batch, vram_val):
                if not project_path:
                    yield "Error: Project directory not set. / プロジェクトディレクトリが設定されていません。"
                    return

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
                    yield "Error: VAE path not set. / VAEのパスが設定されていません。"
                    return

                if not os.path.exists(vae_path_val):
                    yield f"Error: VAE model not found at / 指定されたパスにVAEモデルが見つかりません: {vae_path_val}"
                    return

                config_path = os.path.join(project_path, "dataset_config.toml")
                if not os.path.exists(config_path):
                    yield f"Error: dataset_config.toml not found in {project_path}. Please generate it first. / dataset_config.tomlが {project_path} に見つかりません。先に設定ファイルを生成してください。"
                    return

                script_name = "zimage_cache_latents.py"
                if model == "Qwen-Image":
                    script_name = "qwen_image_cache_latents.py"
                elif model.startswith("Flux.2"):
                     script_name = "flux_2_cache_latents.py"

                script_path = os.path.join("src", "musubi_tuner", script_name)

                cmd = [sys.executable, script_path, "--dataset_config", config_path, "--vae", vae_path_val]

                # Placeholder for argument modification
                if model == "Z-Image-Turbo":
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
                yield f"Starting Latent Caching. Please wait for the first log to appear. / Latentのキャッシュを開始します。最初のログが表示されるまでにしばらくかかります。\nCommand: {command_str}\n\n"

                yield from run_command(command_str)

            def cache_text_encoder(
                project_path, te1_path_val, te2_path_val, vae, model, comfy, w, h, batch, vram_val, fp8_text_encoder
            ):
                if not project_path:
                    yield "Error: Project directory not set. / プロジェクトディレクトリが設定されていません。"
                    return

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
                    yield "Error: Text Encoder 1 path not set. / Text Encoder 1のパスが設定されていません。"
                    return

                if not os.path.exists(te1_path_val):
                    yield f"Error: Text Encoder 1 model not found at / 指定されたパスにText Encoder 1モデルが見つかりません: {te1_path_val}"
                    return

                # Z-Image only uses te1 for now, but keeping te2 in signature if needed later or for other models

                config_path = os.path.join(project_path, "dataset_config.toml")
                if not os.path.exists(config_path):
                    yield f"Error: dataset_config.toml not found in {project_path}. Please generate it first. / dataset_config.tomlが {project_path} に見つかりません。先に設定ファイルを生成してください。"
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
                if model == "Z-Image-Turbo":
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
                yield f"Starting Text Encoder Caching. Please wait for the first log to appear. / Text Encoderのキャッシュを開始します。最初のログが表示されるまでにしばらくかかります。\nCommand: {command_str}\n\n"

                yield from run_command(command_str)

        with gr.Accordion(i18n("acc_training"), open=False, elem_classes=["section-card"]):
            gr.Markdown(i18n("desc_training_basic"))
            training_model_info = gr.Markdown(i18n("desc_training_flux2"))

            with gr.Row():
                set_training_defaults_btn = gr.Button(i18n("btn_rec_params"))
            with gr.Row(elem_classes=["path-row"]):
                dit_path = gr.Textbox(label=i18n("lbl_dit_path"), placeholder=i18n("ph_dit_path"), max_lines=1, scale=8)
                browse_dit = gr.Button(i18n("btn_browse"), scale=1)

            with gr.Row():
                output_name = gr.Textbox(label=i18n("lbl_output_name"), value="my_lora", max_lines=1)

            with gr.Group():
                gr.Markdown(i18n("header_basic_params"))
                with gr.Row():
                    network_dim = gr.Number(label=i18n("lbl_dim"), value=4)
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
                    network_alpha = gr.Number(label=i18n("lbl_network_alpha"), value=1)
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
                with gr.Row(elem_classes=["path-row"]):
                    sample_output_dir = gr.Textbox(
                        label=i18n("lbl_sample_output_dir"),
                        placeholder=i18n("ph_sample_output_dir"),
                        scale=8,
                    )
                    browse_sample_out = gr.Button(i18n("btn_browse"), scale=1)

            with gr.Accordion(i18n("accordion_additional"), open=False):
                gr.Markdown(i18n("desc_additional_args"))
                additional_args = gr.Textbox(label=i18n("lbl_additional_args"), placeholder=i18n("ph_additional_args"))

            with gr.Row(elem_classes=["path-row"]):
                resume_path = gr.Textbox(label=i18n("lbl_resume"), placeholder=i18n("ph_resume"), scale=8)
                browse_resume = gr.Button(i18n("btn_browse"), scale=1)

            training_status = gr.Markdown("")
            with gr.Row():
                start_training_btn = gr.Button(i18n("btn_start_training"), variant="primary", scale=2)
                tensorboard_btn = gr.Button(i18n("btn_tensorboard"), scale=1)

        with gr.Accordion(i18n("acc_post_processing"), open=False, elem_classes=["section-card"]):
            gr.Markdown(i18n("desc_post_proc"))
            with gr.Row():
                set_post_proc_defaults_btn = gr.Button(i18n("btn_set_paths"))
            with gr.Row(elem_classes=["path-row"]):
                input_lora = gr.Textbox(label=i18n("lbl_input_lora"), placeholder=i18n("ph_input_lora"), max_lines=1, scale=8)
                browse_input_lora = gr.Button(i18n("btn_browse"), scale=1)
            with gr.Row(elem_classes=["path-row"]):
                output_comfy_lora = gr.Textbox(label=i18n("lbl_output_comfy"), placeholder=i18n("ph_output_comfy"), max_lines=1, scale=8)
                browse_output_lora = gr.Button(i18n("btn_browse"), scale=1)

            convert_btn = gr.Button(i18n("btn_convert"))
            conversion_log = gr.Textbox(label=i18n("lbl_conversion_log"), lines=5, interactive=False)

        def convert_lora_to_comfy(project_path, input_path, output_path, model, comfy, w, h, batch, vae, te1, te2):
            if not project_path:
                yield "Error: Project directory not set. / プロジェクトディレクトリが設定されていません。"
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
            )

            if not input_path or not output_path:
                yield "Error: Input and Output paths must be specified. / 入力・出力パスを指定してください。"
                return

            if not os.path.exists(input_path):
                yield f"Error: Input file not found at {input_path} / 入力ファイルが見つかりません: {input_path}"
                return

            # Script path
            script_path = os.path.join("src", "musubi_tuner", "networks", "convert_z_image_lora_to_comfy.py")
            if not os.path.exists(script_path):
                yield f"Error: Conversion script not found at {script_path} / 変換スクリプトが見つかりません: {script_path}"
                return

            cmd = [sys.executable, script_path, input_path, output_path]

            command_str = " ".join(cmd)
            yield f"Starting Conversion. / 変換を開始します。\nCommand: {command_str}\n\n"

            yield from run_command(command_str)

        def start_training(
            project_path,
            model,
            dit,
            vae,
            te1,
            output_nm,
            dim,
            lr,
            optimizer_type,
            optimizer_args,
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
            add_args,
            should_sample_images,
            sample_every_n,
            sample_out_dir,
            sample_prompt_val,
            sample_negative_prompt_val,
            sample_w_val,
            sample_h_val,
            resume_from,
            lr_scheduler,  # Added
            lr_scheduler_args,  # Added
        ):
            import shlex

            if not project_path:
                return "Error: Project directory not set. / プロジェクトディレクトリが設定されていません。"
            if not dit:
                return "Error: Base Model / DiT Path not set. / Base Model / DiTのパスが設定されていません。"
            if not os.path.exists(dit):
                return f"Error: Base Model / DiT file not found at {dit} / Base Model / DiTファイルが見つかりません: {dit}"
            if not vae:
                return "Error: VAE Path not set (configure in Preprocessing). / VAEのパスが設定されていません (Preprocessingで設定してください)。"
            if not te1:
                return "Error: Text Encoder 1 Path not set (configure in Preprocessing). / Text Encoder 1のパスが設定されていません (Preprocessingで設定してください)。"

            dataset_config = os.path.join(project_path, "dataset_config.toml")
            if not os.path.exists(dataset_config):
                return "Error: dataset_config.toml not found. Please generate it. / dataset_config.toml が見つかりません。生成してください。"

            output_dir = os.path.join(project_path, "models")
            logging_dir = os.path.join(project_path, "logs")

            # Save settings
            save_project_settings(
                project_path,
                dit_path=dit,
                output_name=output_nm,
                network_dim=dim,
                learning_rate=lr,
                optimizer_type=optimizer_type,
                optimizer_args=optimizer_args,
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

            # Model specific command modification
            if model == "Z-Image-Turbo":
                arch_name = "zimage"
            elif model == "Qwen-Image":
                arch_name = "qwen_image"
            elif model.startswith("Flux.2"):
                arch_name = "flux_2"

            # Construct command for cmd /c to run and then pause
            # We assume 'accelerate' is in the PATH.
            script_path = os.path.join("src", "musubi_tuner", f"{arch_name}_train_network.py")

            # Inner command list - arguments for accelerate launch
            inner_cmd = [
                "accelerate",
                "launch",
                # accelerate args: we don't configure default_config.yaml, so we need to specify all here
                "--num_cpu_threads_per_process",
                "1",
                "--mixed_precision",
                prec,
                "--dynamo_backend=no",
                "--gpu_ids",
                "all",
            ]

            inner_cmd.extend([
                "--machine_rank",
                "0",
                "--main_training_function",
                "main",
                "--num_machines",
                "1",
                "--num_processes",
                "1",
                # script and its args
                script_path,
            ])
            
            # Flux.2 specific arguments for accelerate/script
            if model == "Flux.2 Dev":
                 inner_cmd.extend(["--model_version", "dev"])
            elif model == "Flux.2 Klein (4B)":
                 inner_cmd.extend(["--model_version", "klein-4b"])
            elif model == "Flux.2 Klein Base (4B)":
                 inner_cmd.extend(["--model_version", "klein-base-4b"])

            timestep_sampling = "flux2_shift" if model.startswith("Flux.2") else "shift"

            inner_cmd.extend([
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
                "--network_module",
                f"networks.lora_{arch_name}",
                "--network_dim",
                str(int(dim)),
                "--optimizer_type",
                optimizer_type,
                "--network_alpha",
                str(network_alpha),
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
            ])

            if optimizer_args and optimizer_args.strip():
                try:
                    inner_cmd.append("--optimizer_args")
                    inner_cmd.extend(shlex.split(optimizer_args))
                except Exception as e:
                    return f"Error parsing optimizer args / オプティマイザ引数の解析に失敗しました: {str(e)}"

            if lr_scheduler_args and lr_scheduler_args.strip():
                try:
                    inner_cmd.append("--lr_scheduler_args")
                    inner_cmd.extend(shlex.split(lr_scheduler_args))
                except Exception as e:
                    return f"Error parsing scheduler args / スケジューラ引数の解析に失敗しました: {str(e)}"

            # Sample image generation options
            if should_sample_images:
                sample_prompt_path = os.path.join(project_path, "sample_prompt.txt")
                templates = {
                    # prompt, negative prompt, width, height, flow shift, steps, CFG scale, seed
                    "Qwen-Image": "{prompt} --n {neg} --w {w} --h {h} --fs 2.2 --s 20 --l 4.0 --d 1234",
                    "Z-Image-Turbo": "{prompt} --n {neg} --w {w} --h {h} --fs 3.0 --s 20 --l 5.0 --d 1234",
                    # Flux.2 uses guidance scale; distilled klein uses fewer steps
                    "Flux.2 Dev": "{prompt} --n {neg} --w {w} --h {h} --s 50 --g 4.0 --d 1234",
                    "Flux.2 Klein (4B)": "{prompt} --n {neg} --w {w} --h {h} --s 4 --g 1.0 --d 1234",
                    "Flux.2 Klein Base (4B)": "{prompt} --n {neg} --w {w} --h {h} --s 50 --g 4.0 --d 1234",
                }
                template = templates.get(model, templates["Z-Image-Turbo"])
                prompt_str = (sample_prompt_val or "").replace("\n", " ").strip()
                neg_str = (sample_negative_prompt_val or "").replace("\n", " ").strip()
                try:
                    w_int = int(sample_w_val)
                    h_int = int(sample_h_val)
                except Exception:
                    return "Error: Sample width/height must be integers. / サンプル画像の幅と高さは整数で指定してください。"

                line = template.format(prompt=prompt_str, neg=neg_str, w=w_int, h=h_int)
                try:
                    with open(sample_prompt_path, "w", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception as e:
                    return f"Error writing sample_prompt.txt / sample_prompt.txt の作成に失敗しました: {str(e)}"

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
                if model == "Z-Image-Turbo":
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

            # Model specific command modification
            if model == "Z-Image-Turbo":
                pass
            elif model == "Qwen-Image":
                pass

            # Resume from checkpoint
            if resume_from and resume_from.strip():
                inner_cmd.extend(["--resume", resume_from.strip()])

            # Parse and append additional args
            if add_args:
                try:
                    split_args = shlex.split(add_args)
                    inner_cmd.extend(split_args)
                except Exception as e:
                    return f"Error parsing additional arguments / 追加引数の解析に失敗しました: {str(e)}"

            # Construct the full command string for cmd /c
            # list2cmdline will quote arguments as needed for Windows
            inner_cmd_str = subprocess.list2cmdline(inner_cmd)

            # Chain commands: Run training -> echo message -> pause >nul (hides default message)
            final_cmd_str = f"{inner_cmd_str} & echo. & echo Training finished. Press any key to close this window... 学習が完了しました。このウィンドウを閉じるには任意のキーを押してください。 & pause >nul"

            try:
                # Open in new console window
                flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
                # Pass explicit 'cmd', '/c', string to ensure proper execution
                subprocess.Popen(["cmd", "/c", final_cmd_str], creationflags=flags, shell=False)
                return f"Training started in a new window! / 新しいウィンドウで学習が開始されました！\nCommand: {inner_cmd_str}"
            except Exception as e:
                return f"Error starting training / 学習の開始に失敗しました: {str(e)}"

        def update_model_info(model):
            if model == "Z-Image-Turbo":
                return i18n("desc_training_zimage")
            elif model == "Qwen-Image":
                return i18n("desc_qwen_notes")
            elif model.startswith("Flux.2"):
                return i18n("desc_training_flux2")
            return ""

        # Event wiring moved to end to prevent UnboundLocalError
        init_btn.click(
            fn=init_project,
            inputs=[project_dir],
            outputs=[
                project_status,
                model_arch,
                vram_size,
                comfy_models_dir,
                resolution_w,
                resolution_h,
                batch_size,
                control_directory,
                control_res_w,
                control_res_h,
                no_resize_control,
                toml_preview,
                vae_path,
                text_encoder1_path,
                text_encoder2_path,
                dit_path,
                output_name,
                network_dim,
                learning_rate,
                optimizer_type,
                optimizer_args,
                lr_scheduler,
                lr_scheduler_args,
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
                additional_args,
                sample_images,
                sample_every_n,
                sample_output_dir,
                sample_prompt,
                sample_negative_prompt,
                sample_w,
                sample_h,
                input_lora,
                output_comfy_lora,
            ],
        )

        model_arch.change(fn=update_model_info, inputs=[model_arch], outputs=[training_model_info])

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
            ],
            outputs=[dataset_status, toml_preview],
        )

        validate_models_btn.click(fn=validate_models_dir, inputs=[comfy_models_dir], outputs=[models_status])

        quick_setup_btn.click(
            fn=quick_setup,
            inputs=[project_dir, model_arch, vram_size, comfy_models_dir],
            outputs=[
                resolution_w,
                resolution_h,
                batch_size,
                vae_path,
                text_encoder1_path,
                text_encoder2_path,
                dit_path,
                network_dim,
                learning_rate,
                optimizer_type,
                optimizer_args,
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
                sample_every_n,
                sample_w,
                sample_h,
                sample_output_dir,
                training_model_info,
                quick_status,
            ],
        )

        check_missing_btn.click(
            fn=check_missing,
            inputs=[project_dir, comfy_models_dir, vae_path, text_encoder1_path, dit_path],
            outputs=[quick_status],
        )

        browse_project_dir.click(fn=browse_dir, inputs=[project_dir], outputs=[project_dir])
        browse_comfy_dir.click(fn=browse_dir, inputs=[comfy_models_dir], outputs=[comfy_models_dir])
        browse_control_dir.click(fn=browse_dir, inputs=[control_directory], outputs=[control_directory])

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
            outputs=[resolution_w, resolution_h, batch_size],
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

        set_training_defaults_btn.click(
            fn=set_training_defaults,
            inputs=[project_dir, comfy_models_dir, model_arch, vram_size],
            outputs=[
                dit_path,
                network_dim,
                learning_rate,
                optimizer_type,
                optimizer_args,
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
                sample_every_n,
                sample_w,
                sample_h,
                sample_output_dir,
            ],
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
                network_dim,
                learning_rate,
                optimizer_type,
                optimizer_args,
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
            inputs=[
                preset_name,
                project_dir, model_arch, vram_size, comfy_models_dir,
                resolution_w, resolution_h, batch_size,
                control_directory, control_res_w, control_res_h, no_resize_control,
                vae_path, text_encoder1_path, text_encoder2_path,
                dit_path, output_name, network_dim, learning_rate, optimizer_type, optimizer_args, lr_scheduler, lr_scheduler_args,
                network_alpha, lr_warmup_steps, seed, max_grad_norm,
                num_epochs, save_every_n_epochs,
                discrete_flow_shift, block_swap, use_pinned_memory_for_block_swap, mixed_precision, gradient_checkpointing,
                fp8_scaled, fp8_llm, additional_args,
                sample_images, sample_every_n, sample_prompt, sample_negative_prompt, sample_w, sample_h,
                input_lora, output_comfy_lora
            ],
            outputs=[preset_status]
        )
        
        # Chain save with refresh list
        save_preset_btn.click(fn=refresh_preset_dropdown, outputs=[load_preset_dd])

        load_preset_btn.click(
             fn=load_preset,
             inputs=[load_preset_dd],
             outputs=[
                project_dir, model_arch, vram_size, comfy_models_dir,
                resolution_w, resolution_h, batch_size,
                control_directory, control_res_w, control_res_h, no_resize_control,
                vae_path, text_encoder1_path, text_encoder2_path,
                dit_path, output_name, network_dim, learning_rate, optimizer_type, optimizer_args, lr_scheduler, lr_scheduler_args,
                network_alpha, lr_warmup_steps, seed, max_grad_norm,
                num_epochs, save_every_n_epochs,
                discrete_flow_shift, block_swap, use_pinned_memory_for_block_swap, mixed_precision, gradient_checkpointing,
                fp8_scaled, fp8_llm, additional_args,
                sample_images, sample_every_n, sample_output_dir, sample_prompt, sample_negative_prompt, sample_w, sample_h,
                input_lora, output_comfy_lora
             ]
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
            ],
            outputs=[conversion_log],
        )

    return demo


if __name__ == "__main__":
    demo = construct_ui()
    launch_kwargs = {"i18n": i18n}
    if LAUNCH_SUPPORTS_THEME:
        launch_kwargs["theme"] = gr.themes.Soft()
    if LAUNCH_SUPPORTS_CSS:
        launch_kwargs["css"] = APP_CSS
    demo.launch(**launch_kwargs)
