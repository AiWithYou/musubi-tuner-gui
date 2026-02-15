import os


class ConfigManager:
    def __init__(self):
        self.models = {
            "Z-Image": {
                "resolution": (1024, 1024),
                "vae_subpath": ["vae", "ae.safetensors"],
                "te1_subpath": ["text_encoders", "qwen_3_4b.safetensors"],
                "te2_subpath": None,
                "dit_subpath": ["diffusion_models", "z_image_de_turbo_v1_bf16.safetensors"],
                "training_base": {
                    "learning_rate": 1e-3,
                    "default_num_steps": 500,
                    "save_every_n_epochs": 1,
                    "discrete_flow_shift": 2.0,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                    "fp8_scaled": True,
                    "fp8_llm": True,
                    "training_mode": "LoRA/LoHa/LoKr",
                    "network_type": "LoRA",
                    "network_args": "",
                    "full_bf16": False,
                    "fused_backward_pass": False,
                    "mem_eff_save": False,
                    "block_swap_optimizer_patch_params": False,
                },
                "vram_settings": {
                    "12": {"batch_size": 1, "block_swap": 0, "fp8_scaled": True, "fp8_llm": True},
                    "16": {"batch_size": 2, "block_swap": 0, "fp8_scaled": True, "fp8_llm": False},
                    "24": {"batch_size": 1, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                    "32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                    ">32": {"batch_size": 8, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                },
            },
            "Qwen-Image": {
                "resolution": (1328, 1328),
                "vae_subpath": ["vae", "qwen_image_vae.safetensors"],
                "te1_subpath": ["text_encoders", "qwen_2.5_vl_7b.safetensors"],
                "te2_subpath": None,
                "dit_subpath": ["diffusion_models", "qwen_image_bf16.safetensors"],
                "training_base": {
                    "learning_rate": 1e-3,
                    "default_num_steps": 1000,
                    "save_every_n_epochs": 1,
                    "discrete_flow_shift": 2.2,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                    "fp8_scaled": True,
                    "fp8_llm": True,
                    "training_mode": "LoRA/LoHa/LoKr",
                    "network_type": "LoRA",
                    "network_args": "",
                    "full_bf16": False,
                    "fused_backward_pass": False,
                    "mem_eff_save": False,
                    "block_swap_optimizer_patch_params": False,
                },
                "vram_settings": {
                    "12": {"batch_size": 1, "block_swap": 46, "fp8_scaled": True, "fp8_llm": True},
                    "16": {"batch_size": 1, "block_swap": 34, "fp8_scaled": True, "fp8_llm": True},
                    "24": {"batch_size": 1, "block_swap": 10, "fp8_scaled": True, "fp8_llm": False},
                    "32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": True, "fp8_llm": False},
                    ">32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                },
            },
            "Flux.2 Dev": {
                "resolution": (1024, 1024),
                "vae_subpath": ["vae", "ae.safetensors"],
                # Text encoder for dev is split; leave empty to avoid wrong default.
                "te1_subpath": None,
                "te2_subpath": None,
                "dit_subpath": ["diffusion_models", "flux2-dev.safetensors"],
                "training_base": {
                    "learning_rate": 1e-4,
                    "network_dim": 32,
                    "default_num_steps": 1000,
                    "save_every_n_epochs": 1,
                    # Flux 2 doesn't use discrete flow shift in the same way or uses None defaults often, 
                    # but code often respects passed value. Using default often seen in args.
                    "discrete_flow_shift": 1.0, 
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                    "fp8_scaled": True,
                    "fp8_llm": False,  # Mistral uses fp8_text_encoder; keep disabled by default
                    "training_mode": "LoRA/LoHa/LoKr",
                    "network_type": "LoRA",
                    "network_args": "",
                    "full_bf16": False,
                    "fused_backward_pass": False,
                    "mem_eff_save": False,
                    "block_swap_optimizer_patch_params": False,
                },
                "vram_settings": {
                    "12": {"batch_size": 1, "block_swap": 20, "fp8_scaled": True, "fp8_llm": False},
                    "16": {"batch_size": 1, "block_swap": 10, "fp8_scaled": True, "fp8_llm": False},
                    "24": {"batch_size": 1, "block_swap": 0, "fp8_scaled": True, "fp8_llm": False},
                    "32": {"batch_size": 1, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                    ">32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                },
            },
            "Flux.2 Klein (4B)": {
                "resolution": (1024, 1024),
                "vae_subpath": ["vae", "ae.safetensors"],
                "te1_subpath": ["text_encoders", "qwen_3_4b.safetensors"],
                "te2_subpath": None,
                "dit_subpath": ["diffusion_models", "flux2-klein-4b.safetensors"],
                "training_base": {
                    "learning_rate": 1e-4,
                    "network_dim": 32,
                    "default_num_steps": 1000,
                    "save_every_n_epochs": 1,
                    "discrete_flow_shift": 1.0,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                    "fp8_scaled": True,
                    "fp8_llm": False,
                    "training_mode": "LoRA/LoHa/LoKr",
                    "network_type": "LoRA",
                    "network_args": "",
                    "full_bf16": False,
                    "fused_backward_pass": False,
                    "mem_eff_save": False,
                    "block_swap_optimizer_patch_params": False,
                },
                "vram_settings": {
                    "12": {"batch_size": 1, "block_swap": 10, "fp8_scaled": True, "fp8_llm": True},
                    "16": {"batch_size": 1, "block_swap": 0, "fp8_scaled": True, "fp8_llm": False},
                    "24": {"batch_size": 2, "block_swap": 0, "fp8_scaled": True, "fp8_llm": False},
                    "32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                    ">32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                },
            },
            "Flux.2 Klein Base (4B)": {
                "resolution": (1024, 1024),
                "vae_subpath": ["vae", "ae.safetensors"],
                "te1_subpath": ["text_encoders", "qwen_3_4b.safetensors"],
                "te2_subpath": None,
                "dit_subpath": ["diffusion_models", "flux2-klein-base-4b.safetensors"],
                "training_base": {
                    "learning_rate": 1e-4,
                    "network_dim": 32,
                    "default_num_steps": 1000,
                    "save_every_n_epochs": 1,
                    "discrete_flow_shift": 1.0,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                    "fp8_scaled": True,
                    "fp8_llm": False,
                    "training_mode": "LoRA/LoHa/LoKr",
                    "network_type": "LoRA",
                    "network_args": "",
                    "full_bf16": False,
                    "fused_backward_pass": False,
                    "mem_eff_save": False,
                    "block_swap_optimizer_patch_params": False,
                },
                "vram_settings": {
                    "12": {"batch_size": 1, "block_swap": 10, "fp8_scaled": True, "fp8_llm": True},
                    "16": {"batch_size": 1, "block_swap": 0, "fp8_scaled": True, "fp8_llm": False},
                    "24": {"batch_size": 2, "block_swap": 0, "fp8_scaled": True, "fp8_llm": False},
                    "32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                    ">32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                },
            },
        }
        # Backward compatible alias for older presets
        self.models["Z-Image-Turbo"] = self.models["Z-Image"]

    def get_resolution(self, model_name):
        return self.models.get(model_name, {}).get("resolution", (1024, 1024))

    def get_batch_size(self, model_name, vram_size):
        # Default to "16" if vram_size is not provided or invalid, as a safe middle ground
        if not vram_size:
            vram_size = "16"

        vram_conf = self.models.get(model_name, {}).get("vram_settings", {}).get(vram_size, {})
        return vram_conf.get("batch_size", 1)

    def get_preprocessing_paths(self, model_name, comfy_models_dir):
        conf = self.models.get(model_name, {})
        vae = conf.get("vae_subpath")
        te1 = conf.get("te1_subpath")
        te2 = conf.get("te2_subpath")

        def join_path(subpath):
            if subpath and comfy_models_dir:
                return os.path.join(comfy_models_dir, *subpath)
            return ""

        return join_path(vae), join_path(te1), join_path(te2)

    def get_training_defaults(self, model_name, vram_size, comfy_models_dir):
        conf = self.models.get(model_name, {})
        base = conf.get("training_base", {}).copy()

        if not vram_size:
            vram_size = "16"

        # Merge VRAM settings
        vram_conf = conf.get("vram_settings", {}).get(vram_size, {})
        # Only take relevant keys for training params (block_swap), ignore batch_size as it's for dataset
        if "block_swap" in vram_conf:
            base["block_swap"] = vram_conf["block_swap"]
        if "fp8_llm" in vram_conf:
            base["fp8_llm"] = vram_conf["fp8_llm"]
        if "fp8_scaled" in vram_conf:
            base["fp8_scaled"] = vram_conf["fp8_scaled"]

        # DiT path
        dit_sub = conf.get("dit_subpath")
        if dit_sub and comfy_models_dir:
            base["dit_path"] = os.path.join(comfy_models_dir, *dit_sub)
        else:
            base["dit_path"] = ""

        return base
