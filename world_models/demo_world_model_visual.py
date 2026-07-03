"""
Demo MiniWorld World Model - 2 cột độc lập
=========================================

Output:
1. real_video.mp4              : video thật từ replay buffer
2. world_model_prediction.mp4  : video dự đoán từ World Model
3. demo_report.html            : hiển thị 2 video thành 2 cột độc lập

Cài thêm:
pip install imageio imageio-ffmpeg

CLI:
python demo_world_model_visual.py --run_dir runs_hallway_200k_light --benchmark_py miniworld_wm_benchmark_final.py --model dreamer --video_frames 100 --video_scale 5 --video_fps 8
"""

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# =========================================================
# 1. Load file benchmark gốc
# =========================================================

def load_benchmark_module(benchmark_py: str):
    benchmark_py = Path(benchmark_py)

    if not benchmark_py.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file benchmark: {benchmark_py}"
        )

    spec = importlib.util.spec_from_file_location("wm_benchmark", benchmark_py)
    wm = importlib.util.module_from_spec(spec)
    sys.modules["wm_benchmark"] = wm
    spec.loader.exec_module(wm)

    return wm


# =========================================================
# 2. Video utilities
# =========================================================

def get_nearest_resample():
    try:
        return Image.Resampling.NEAREST
    except AttributeError:
        return Image.NEAREST


def upscale_np_img(img_np, scale=5):
    """
    Phóng to ảnh numpy [H, W, 3].
    Không thêm chữ vào frame để video sạch hơn.
    """
    img = Image.fromarray(img_np).convert("RGB")

    if scale <= 1:
        return img

    w, h = img.size

    return img.resize(
        (w * scale, h * scale),
        get_nearest_resample()
    )


def tensor_img_to_uint8(x):
    """
    Tensor ảnh [3, H, W], giá trị [0,1]
    -> numpy ảnh [H, W, 3], uint8.
    """
    x = x.detach().cpu().clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()

    return (x * 255).astype(np.uint8)


def write_mp4_from_np_images(images_np, out_path, fps=8, scale=5):
    """
    Ghi list/array ảnh numpy [T, H, W, 3] thành MP4.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "Bạn cần cài thêm:\n"
            "pip install imageio imageio-ffmpeg"
        )

    frames = []

    for img_np in images_np:
        frame = upscale_np_img(img_np, scale=scale)
        frames.append(np.asarray(frame))

    with imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def write_mp4_from_tensor_images(images_tensor, out_path, fps=8, scale=5):
    """
    Ghi tensor ảnh [T, 3, H, W] thành MP4.
    """
    images_np = []

    for i in range(images_tensor.shape[0]):
        img_np = tensor_img_to_uint8(images_tensor[i])
        images_np.append(img_np)

    write_mp4_from_np_images(
        images_np=images_np,
        out_path=out_path,
        fps=fps,
        scale=scale,
    )


# =========================================================
# 3. Chọn đoạn video từ replay buffer
# =========================================================

def select_episode_and_sequence(
    episodes,
    episode_idx=-1,
    start=0,
    video_frames=100,
):
    """
    Chọn một đoạn dài từ replay buffer.

    episode_idx = -1: tự chọn episode dài nhất.
    """
    if len(episodes) == 0:
        raise ValueError("Replay buffer rỗng.")

    if episode_idx < 0:
        lengths = [len(ep["actions"]) for ep in episodes]
        episode_idx = int(np.argmax(lengths))
    else:
        episode_idx = min(episode_idx, len(episodes) - 1)

    ep = episodes[episode_idx]
    total_actions = len(ep["actions"])

    if total_actions <= 1:
        raise ValueError("Episode quá ngắn.")

    max_start = max(0, total_actions - 1)
    start = max(0, min(start, max_start))

    available = total_actions - start
    frames = min(video_frames, available)

    if frames < video_frames:
        print(
            f"[Warning] Chỉ lấy được {frames} frame "
            f"thay vì {video_frames} frame."
        )

    obs_uint8 = ep["obs"][start:start + frames + 1]
    actions_np = ep["actions"][start:start + frames]

    return episode_idx, start, obs_uint8, actions_np


# =========================================================
# 4. Decode latent thành ảnh
# =========================================================

@torch.no_grad()
def decode_latents_in_chunks(model, pred_z, chunk_size=128):
    """
    Decode latent theo chunk để tránh tốn VRAM.
    pred_z: [B, L, D]
    return: [B, L, 3, H, W]
    """
    B, L, D = pred_z.shape
    flat_z = pred_z.reshape(B * L, D)

    imgs = []

    for i in range(0, flat_z.shape[0], chunk_size):
        z_chunk = flat_z[i:i + chunk_size]
        img_chunk = model.decoder(z_chunk)
        imgs.append(img_chunk)

    imgs = torch.cat(imgs, dim=0)
    imgs = imgs.reshape(B, L, imgs.shape[1], imgs.shape[2], imgs.shape[3])

    return imgs


# =========================================================
# 5. Dự đoán frame bằng World Model
# =========================================================

@torch.no_grad()
def teacher_forced_prediction_images(model, obs, actions, n_actions):
    """
    Dự đoán nhiều frame bằng teacher-forced mode.

    Ý nghĩa:
    - Mỗi bước dùng latent thật z_t.
    - World Model dự đoán z_{t+1}.
    - Decode z_{t+1} thành ảnh.
    - Cách này ổn định hơn khi demo video dài.

    obs: [1, L+1, 3, H, W]
    actions: [1, L]

    return:
    pred_imgs: [L, 3, H, W]
    """
    model.eval()

    B, Lp1, C, H, W = obs.shape
    L = Lp1 - 1

    true_z = model.encode_mu(obs.reshape(B * Lp1, C, H, W))
    true_z = true_z.reshape(B, Lp1, -1)

    pred_zs = []

    # Transformer World Model
    if hasattr(model, "transformer"):
        for t in range(L):
            start = max(0, t + 1 - model.context_len)

            ctx_z = true_z[:, start:t + 1]
            ctx_actions = actions[:, start:t + 1]

            T = ctx_z.shape[1]

            token = (
                model.z_proj(ctx_z)
                + model.a_embed(ctx_actions)
                + model.pos_embed[:, :T]
            )

            mask = model.causal_mask(T, obs.device)
            h = model.transformer(token, mask=mask)

            z_next = model.next_z_head(h[:, -1])
            pred_zs.append(z_next)

    # PlaNet / Dreamer RSSM
    elif hasattr(model, "prior_mu"):
        h = torch.zeros(B, model.deter_dim, device=obs.device)

        for t in range(L):
            z_t = true_z[:, t]
            a_t = F.one_hot(actions[:, t], n_actions).float()

            h = model.rnn(torch.cat([z_t, a_t], dim=-1), h)
            z_next = model.prior_mu(h)

            pred_zs.append(z_next)

    # Ha-style World Model
    else:
        h = torch.zeros(B, model.hidden_dim, device=obs.device)

        for t in range(L):
            z_t = true_z[:, t]
            a_t = F.one_hot(actions[:, t], n_actions).float()

            h = model.rnn(torch.cat([z_t, a_t], dim=-1), h)
            z_next = model.next_z_head(h)

            pred_zs.append(z_next)

    pred_z = torch.stack(pred_zs, dim=1)
    pred_imgs = decode_latents_in_chunks(model, pred_z)

    return pred_imgs[0]


# =========================================================
# 6. HTML 2 cột độc lập
# =========================================================

def make_html_report(
    out_path,
    model_name,
    real_video,
    pred_video,
    config,
    video_frames,
):
    config_rows = ""

    for k, v in config.items():
        config_rows += f"<tr><td>{k}</td><td>{v}</td></tr>\n"

    html = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>World Model Demo - Two Columns</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 32px;
            background: #f7f9fc;
            color: #111827;
        }}

        h1 {{
            margin-bottom: 8px;
        }}

        .subtitle {{
            color: #4b5563;
            margin-bottom: 24px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            align-items: start;
        }}

        .card {{
            background: white;
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        video {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid #d1d5db;
            background: black;
        }}

        .label {{
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 12px;
            text-align: center;
        }}

        .note {{
            margin-top: 20px;
            background: white;
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 12px;
        }}

        td, th {{
            border: 1px solid #ddd;
            padding: 8px;
        }}

        th {{
            background: #e5e7eb;
        }}

        code {{
            background: #eef2ff;
            padding: 2px 6px;
            border-radius: 4px;
        }}
    </style>
</head>

<body>

<h1>MiniWorld World Model Demo</h1>

<div class="subtitle">
    Model: <code>{model_name}</code> |
    Số frame: <code>{video_frames}</code>
</div>

<div class="grid">

    <div class="card">
        <div class="label">Video thật từ replay buffer</div>
        <video controls autoplay loop muted>
            <source src="{real_video.name}" type="video/mp4">
        </video>
    </div>

    <div class="card">
        <div class="label">Ảnh dự đoán từ World Model</div>
        <video controls autoplay loop muted>
            <source src="{pred_video.name}" type="video/mp4">
        </video>
    </div>

</div>

<div class="note">
    <h2>Ý nghĩa demo</h2>
    <p>
        Cột trái là chuỗi quan sát thật mà agent thu được từ môi trường MiniWorld.
        Cột phải là chuỗi ảnh được tái tạo/dự đoán từ World Model.
        Nếu cột phải càng giống cột trái thì World Model học động lực môi trường càng tốt.
    </p>
</div>

<div class="note">
    <h2>Cấu hình chạy</h2>
    <table>
        <tr>
            <th>Tham số</th>
            <th>Giá trị</th>
        </tr>
        {config_rows}
    </table>
</div>

</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# =========================================================
# 7. Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Folder chứa kết quả train, ví dụ: runs_hallway_200k_light",
    )

    parser.add_argument(
        "--benchmark_py",
        type=str,
        default="miniworld_wm_benchmark_final.py",
        help="File code benchmark gốc.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        help="auto / ha / planet / dreamer / transformer",
    )

    parser.add_argument(
        "--video_frames",
        type=int,
        default=100,
        help="Số frame video.",
    )

    parser.add_argument(
        "--video_fps",
        type=int,
        default=8,
        help="FPS video.",
    )

    parser.add_argument(
        "--video_scale",
        type=int,
        default=5,
        help="Độ phóng to ảnh. 5 nghĩa là 64x64 thành 320x320.",
    )

    parser.add_argument(
        "--episode_idx",
        type=int,
        default=-1,
        help="-1 là tự chọn episode dài nhất.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Frame bắt đầu.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto / cpu / cuda",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy run_dir: {run_dir}")

    demo_dir = run_dir / "demo_outputs"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = run_dir / "config.json"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    model_seq_len = int(config.get("seq_len", 8))
    env_id = config.get("env_id", "MiniWorld-Hallway-v0")

    # Load benchmark
    wm = load_benchmark_module(args.benchmark_py)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load replay buffer
    buffer_files = sorted(run_dir.glob("buffer_*.pkl"))

    if len(buffer_files) == 0:
        raise FileNotFoundError(
            f"Không tìm thấy replay buffer .pkl trong {run_dir}"
        )

    buffer_path = buffer_files[0]
    print(f"Loading replay buffer: {buffer_path}")

    with open(buffer_path, "rb") as f:
        episodes = pickle.load(f)

    episode_idx, start, obs_uint8, actions_np = select_episode_and_sequence(
        episodes=episodes,
        episode_idx=args.episode_idx,
        start=args.start,
        video_frames=args.video_frames,
    )

    real_video_frames = len(actions_np)

    print(f"Selected episode_idx: {episode_idx}")
    print(f"Start frame: {start}")
    print(f"Video frames: {real_video_frames}")

    # Tạo video thật
    real_video = demo_dir / "real_video.mp4"

    write_mp4_from_np_images(
        images_np=obs_uint8[1:],
        out_path=real_video,
        fps=args.video_fps,
        scale=args.video_scale,
    )

    print(f"Saved real video: {real_video}")

    # Chọn checkpoint
    ckpt_files = sorted(run_dir.glob("*_best.pt"))

    if len(ckpt_files) == 0:
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint *_best.pt trong {run_dir}"
        )

    if args.model == "auto":
        priority = ["dreamer", "planet", "transformer", "ha"]
        chosen = None

        for p in priority:
            for ckpt in ckpt_files:
                if ckpt.name.startswith(p + "_"):
                    chosen = ckpt
                    break

            if chosen is not None:
                break

        if chosen is None:
            chosen = ckpt_files[0]

        model_name = chosen.name.replace("_best.pt", "")

    else:
        model_name = args.model.lower()
        chosen = run_dir / f"{model_name}_best.pt"

        if not chosen.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {chosen}"
            )

    print(f"Loading model: {model_name}")
    print(f"Checkpoint: {chosen}")

    # Lấy số action
    try:
        env = wm.make_env(env_id)
        n_actions = int(env.action_space.n)
        env.close()
    except Exception:
        print("Không tạo được env, fallback n_actions = 3")
        n_actions = 3

    # Build model
    model = wm.build_model(
        model_name,
        n_actions=n_actions,
        seq_len=model_seq_len,
    )

    model.to(device)

    try:
        state = torch.load(
            chosen,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            chosen,
            map_location=device,
        )

    model.load_state_dict(state)
    model.eval()

    # Chuẩn bị tensor
    obs_t = wm.obs_uint8_to_float_tensor(obs_uint8)
    obs_t = obs_t.unsqueeze(0).to(device)

    actions_t = torch.from_numpy(actions_np)
    actions_t = actions_t.long().unsqueeze(0).to(device)

    # Dự đoán ảnh từ World Model
    pred_imgs = teacher_forced_prediction_images(
        model=model,
        obs=obs_t,
        actions=actions_t,
        n_actions=n_actions,
    )

    pred_video = demo_dir / "world_model_prediction.mp4"

    write_mp4_from_tensor_images(
        images_tensor=pred_imgs,
        out_path=pred_video,
        fps=args.video_fps,
        scale=args.video_scale,
    )

    print(f"Saved World Model prediction video: {pred_video}")

    # Tạo HTML 2 cột
    report_path = demo_dir / "demo_report.html"

    make_html_report(
        out_path=report_path,
        model_name=model_name,
        real_video=real_video,
        pred_video=pred_video,
        config=config,
        video_frames=real_video_frames,
    )

    print(f"Saved HTML report: {report_path}")

    print("\nDONE.")
    print("Mở file sau bằng trình duyệt:")
    print(report_path)


if __name__ == "__main__":
    main()