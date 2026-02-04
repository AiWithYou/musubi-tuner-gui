# Musubi Tuner GUI

## 概要
Musubi Tuner GUI は、LoRA 学習を行うための GUI フロントエンドです。  
この README は **GUI の使い方に限定**しています。

## Overview
Musubi Tuner GUI is a GUI frontend for LoRA training.  
This README focuses on **GUI usage only**.

## フォルダ構成
`musubi/` 配下で、`gui/` と `musubi-tuner/` が並んでいる前提です。

```text
musubi/
├─ gui/              # GUI 本体 (gui.py, start_gui.bat)
└─ musubi-tuner/     # 本体リポジトリ
```

## Folder Structure
Expected layout under `musubi/`:

```text
musubi/
├─ gui/              # GUI app (gui.py, start_gui.bat)
└─ musubi-tuner/     # Main repository
```

## 前提条件
- Windows 環境
- `musubi-tuner` 側のセットアップが完了していること
- ComfyUI の必要モデルが配置済みであること

※ Python / Git / uv のインストールや `musubi-tuner` 自体の導入手順は、`musubi-tuner` の README を参照してください。

## Prerequisites
- Windows environment
- `musubi-tuner` setup is already completed
- Required ComfyUI models are already prepared

For Python/Git/uv installation and base repository setup, use the `musubi-tuner` README.

## GUI の起動
**推奨:** `gui/` フォルダで `start_gui.bat` を実行

手動起動:

### CUDA 12.4
```powershell
uv run --extra cu124 --extra gui python src/musubi_tuner/gui/gui.py
```

### CUDA 12.8
```powershell
uv run --extra cu128 --extra gui python src/musubi_tuner/gui/gui.py
```

初回起動は依存解決で時間がかかる場合があります。

## Launching the GUI
**Recommended:** run `start_gui.bat` in the `gui/` folder.

Manual launch:

### CUDA 12.4
```powershell
uv run --extra cu124 --extra gui python src/musubi_tuner/gui/gui.py
```

### CUDA 12.8
```powershell
uv run --extra cu128 --extra gui python src/musubi_tuner/gui/gui.py
```

The first launch may take time due to dependency resolution.

## 基本ワークフロー
1. `Project Directory` を設定して初期化
2. `Model Architecture / VRAM / ComfyUI models` を設定
3. `Generate Dataset Config` を実行
4. `Preprocessing` でキャッシュを作成
5. `Training` を開始

## Basic Workflow
1. Set `Project Directory` and initialize
2. Configure `Model Architecture / VRAM / ComfyUI models`
3. Run `Generate Dataset Config`
4. Run preprocessing cache steps
5. Start training

## トラブルシューティング
- `ComfyUI models` 検証エラー:
  - `vae` / `text_encoders` / `diffusion_models` の配置を確認
- `dataset_config.toml` がない:
  - `Generate Dataset Config` を実行
- VRAM 不足:
  - 解像度・バッチサイズを下げる / Block Swap を利用

## Troubleshooting
- `ComfyUI models` validation fails:
  - Ensure `vae` / `text_encoders` / `diffusion_models` are present
- `dataset_config.toml` missing:
  - Run `Generate Dataset Config`
- Out of VRAM:
  - Lower resolution/batch size or use Block Swap

## 参考リンク
- [GUI User Guide (English)](./gui.md)

## References
- [GUI User Guide (English)](./gui.md)
