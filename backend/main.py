import os
import uuid
import shutil
import subprocess
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WORK_DIR = Path("/tmp/metaclean")
WORK_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok"}

# ═══════════════════════════════════════════════════════════════════════════
# VIDEO ATTACK — MAIN ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/attack/video")
async def attack_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(None),
    cover: UploadFile = File(None),
    level: int = Form(1),
    tiktok: bool = Form(False),
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        # Save uploaded files
        input_path = job_dir / f"input{_ext(video.filename)}"
        with open(input_path, "wb") as f:
            f.write(await video.read())

        audio_path = None
        if audio:
            audio_path = job_dir / f"audio{_ext(audio.filename)}"
            with open(audio_path, "wb") as f:
                f.write(await audio.read())

        cover_path = None
        if cover:
            cover_path = job_dir / f"cover{_ext(cover.filename)}"
            with open(cover_path, "wb") as f:
                f.write(await cover.read())

        output_path = job_dir / "output.mp4"

        # Process
        process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            audio_path=str(audio_path) if audio_path else None,
            cover_path=str(cover_path) if cover_path else None,
            level=max(1, min(3, level)),
            tiktok=tiktok,
            job_dir=str(job_dir),
        )

        return FileResponse(
            str(output_path),
            media_type="video/mp4",
            filename=f"adv_{video.filename.rsplit('.', 1)[0]}.mp4",
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Cleanup after response is sent
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
        except:
            pass


def _ext(filename):
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".mp4"


# ═══════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def process_video(input_path, output_path, audio_path, cover_path, level, tiktok, job_dir):
    import cv2

    frames_dir = os.path.join(job_dir, "frames")
    processed_dir = os.path.join(job_dir, "processed")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # ── Get video info ──
    info = get_video_info(input_path)
    fps = info["fps"]
    width = info["width"]
    height = info["height"]
    duration = info["duration"]

    # ── Speed variation (99-101%) ──
    speed_factor = 0.99 + np.random.random() * 0.02

    # ── Extract frames ──
    extract_frames(input_path, frames_dir, fps)

    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".png")],
        key=lambda x: int(x.split(".")[0].split("_")[-1]) if "_" in x else 0,
    )

    if not frame_files:
        raise ValueError("No frames extracted from video")

    # ── Load cover image ──
    if cover_path and os.path.exists(cover_path):
        cover_img = cv2.imread(cover_path)
        cover_img = cv2.resize(cover_img, (width, height))
    else:
        # Use first frame as cover
        cover_img = cv2.imread(os.path.join(frames_dir, frame_files[0]))

    # ── Generate intro frames (0.5s) ──
    intro_count = max(1, int(fps * 0.5))
    frame_idx = 0
    for i in range(intro_count):
        intro = generate_intro_frame(width, height)
        cv2.imwrite(os.path.join(processed_dir, f"frame_{frame_idx:06d}.png"), intro)
        frame_idx += 1

    # ── Cover as first content frame (with transforms) ──
    cover_processed = cover_img.copy()
    cover_processed = apply_crop_and_scale(cover_processed, width, height)
    cover_processed = apply_color_shift(cover_processed)
    cover_processed = apply_noise(cover_processed, seed=42, eps=4, tiktok=tiktok, frame_idx=0)
    cv2.imwrite(os.path.join(processed_dir, f"frame_{frame_idx:06d}.png"), cover_processed)
    frame_idx += 1

    # ── Process each original frame ──
    for i, fname in enumerate(frame_files):
        frame = cv2.imread(os.path.join(frames_dir, fname))
        if frame is None:
            continue

        # Geometric transforms
        frame = apply_crop_and_scale(frame, width, height)
        frame = apply_color_shift(frame)

        # Level >= 2: overlay cover at low opacity
        if level >= 2:
            frame = apply_overlay(frame, cover_img, alpha=0.06)

        # Noise (adapted per frame)
        seed = 1000 + i
        eps = 4
        frame = apply_noise(frame, seed=seed, eps=eps, tiktok=tiktok, frame_idx=i)

        # Invisible pixels every 3 frames
        if i % 3 == 0:
            frame = apply_invisible_pixels(frame, seed=seed)

        # Level 3: color flashes
        if level == 3 and i % 15 == 0:
            frame = apply_color_flash(frame)

        cv2.imwrite(os.path.join(processed_dir, f"frame_{frame_idx:06d}.png"), frame)
        frame_idx += 1

    # ── Generate outro frames (0.5s) ──
    for i in range(intro_count):
        outro = generate_intro_frame(width, height)
        cv2.imwrite(os.path.join(processed_dir, f"frame_{frame_idx:06d}.png"), outro)
        frame_idx += 1

    # ── Extension frames for level 2 & 3 ──
    target_duration = {1: 0, 2: 300, 3: 600}.get(level, 0)  # seconds
    current_duration = frame_idx / fps

    if level >= 2 and target_duration > current_duration:
        extension_frames = int((target_duration - current_duration) * fps)
        for i in range(extension_frames):
            ext_frame = cover_img.copy()
            ext_frame = apply_noise(ext_frame, seed=50000 + i, eps=3, tiktok=False, frame_idx=i)

            if level == 3 and i % 8 == 0:
                ext_frame = apply_color_flash(ext_frame)

            cv2.imwrite(os.path.join(processed_dir, f"frame_{frame_idx:06d}.png"), ext_frame)
            frame_idx += 1

    # ── Encode final video with FFmpeg ──
    effective_fps = fps * speed_factor
    temp_video = os.path.join(job_dir, "temp_video.mp4")

    # Encode processed frames to video
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(round(effective_fps, 2)),
        "-i", os.path.join(processed_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-map_metadata", "-1",
        "-fflags", "+bitexact",
        temp_video,
    ]
    run_ffmpeg(ffmpeg_cmd)

    # ── Handle audio ──
    if audio_path and os.path.exists(audio_path):
        # Use external audio (anti-transcription)
        merge_audio(temp_video, audio_path, output_path, target_duration if level >= 2 else duration)
    else:
        # Re-encode original audio (strip fingerprint)
        original_audio = os.path.join(job_dir, "original_audio.aac")
        extract_audio(input_path, original_audio)
        if os.path.exists(original_audio):
            merge_audio(temp_video, original_audio, output_path, target_duration if level >= 2 else duration)
        else:
            shutil.copy2(temp_video, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# FRAME MANIPULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def apply_crop_and_scale(frame, target_w, target_h):
    import cv2
    h, w = frame.shape[:2]
    crop_pct = 0.01 + np.random.random() * 0.02  # 1-3%
    cx = int(w * crop_pct)
    cy = int(h * crop_pct)
    cropped = frame[cy:h - cy, cx:w - cx]
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def apply_micro_rotation(frame):
    import cv2
    h, w = frame.shape[:2]
    angle = (0.3 + np.random.random() * 0.7) * (1 if np.random.random() > 0.5 else -1)
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_color_shift(frame):
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = (np.random.random() - 0.5) * 10    # ±5
    sat_mult = 0.97 + np.random.random() * 0.06     # 97-103%
    val_mult = 0.97 + np.random.random() * 0.06     # 97-103%
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_mult, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_noise(frame, seed, eps, tiktok, frame_idx):
    h, w = frame.shape[:2]
    block_size = 8 if h * w > 2_000_000 else 6 if h * w > 1_000_000 else 4
    rng = np.random.RandomState(seed)
    tiktok_mult = 1.5 if tiktok and frame_idx % 2 == 1 else 1.0
    tiktok_invert = tiktok and frame_idx % 2 == 1

    result = frame.astype(np.int16)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            direction = 1 if ((x + y) >> 2) & 1 else -1
            magnitude = int((eps + rng.randint(-1, 2)) * direction * tiktok_mult)
            ey = min(y + block_size, h)
            ex = min(x + block_size, w)
            if tiktok_invert:
                result[y:ey, x:ex, 0] -= magnitude
                result[y:ey, x:ex, 1] -= magnitude
                result[y:ey, x:ex, 2] += magnitude
            else:
                result[y:ey, x:ex, 0] += magnitude
                result[y:ey, x:ex, 1] += magnitude
                result[y:ey, x:ex, 2] -= magnitude

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_invisible_pixels(frame, seed):
    rng = np.random.RandomState(seed)
    h, w = frame.shape[:2]
    result = frame.copy()
    for _ in range(6):
        px = rng.randint(0, w)
        py = rng.randint(0, h)
        c = 0 if rng.randint(0, 2) == 0 else 255
        alpha = 0.01
        result[py, px] = (
            np.clip(result[py, px].astype(np.float32) * (1 - alpha) + c * alpha, 0, 255)
        ).astype(np.uint8)
    return result


def apply_color_flash(frame):
    overlay = np.full_like(frame, [
        np.random.randint(0, 256),
        np.random.randint(0, 256),
        np.random.randint(0, 256),
    ], dtype=np.uint8)
    alpha = 0.08 if np.random.random() > 0.5 else 0.12
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def apply_overlay(frame, overlay_img, alpha=0.06):
    import cv2
    overlay_resized = cv2.resize(overlay_img, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1 - alpha, overlay_resized, alpha, 0)


def generate_intro_frame(width, height):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Dark gradient with random tones
    r1, g1, b1 = np.random.randint(0, 40, 3)
    r2, g2, b2 = np.random.randint(0, 60, 3)
    for y in range(height):
        t = y / height
        frame[y, :, 0] = int(b1 * (1 - t) + b2 * t)
        frame[y, :, 1] = int(g1 * (1 - t) + g2 * t)
        frame[y, :, 2] = int(r1 * (1 - t) + r2 * t)
    # Add subtle noise
    noise = np.random.randint(-4, 5, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# FFMPEG HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_video_info(path):
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)

    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not video_stream:
        raise ValueError("No video stream found")

    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
    fps = min(fps, 60)  # cap at 60fps

    return {
        "fps": fps,
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "duration": float(data.get("format", {}).get("duration", 30)),
    }


def extract_frames(input_path, output_dir, fps):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%06d.png"),
    ]
    run_ffmpeg(cmd)


def extract_audio(input_path, output_path):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "aac", "-b:a", "128k",
        "-map_metadata", "-1",
        output_path,
    ]
    try:
        run_ffmpeg(cmd)
    except:
        pass  # Video might have no audio


def merge_audio(video_path, audio_path, output_path, target_duration):
    # Loop audio if video is longer (extension frames)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-stream_loop", "-1", "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        "-map_metadata", "-1",
        "-fflags", "+bitexact",
        "-movflags", "+faststart",
        output_path,
    ]
    run_ffmpeg(cmd)


def run_ffmpeg(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr[-500:]}")


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE ATTACK (optional endpoint)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/attack/image")
async def attack_image(
    image: UploadFile = File(...),
    eps: int = Form(4),
    iterations: int = Form(20),
):
    import cv2

    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        input_path = job_dir / f"input{_ext(image.filename)}"
        with open(input_path, "wb") as f:
            f.write(await image.read())

        img = cv2.imread(str(input_path))
        h, w = img.shape[:2]

        # Geometric transforms
        img = apply_crop_and_scale(img, w, h)
        img = apply_micro_rotation(img)
        img = apply_color_shift(img)

        # PGD-style noise with edge awareness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge_map = (edge_map / edge_map.max() * 255).astype(np.float32) if edge_map.max() > 0 else edge_map

        result = img.astype(np.float32)
        orig = img.astype(np.float32)
        for _ in range(iterations):
            sensitivity = edge_map / 255.0
            noise = (np.random.random(img.shape[:2]).astype(np.float32) - 0.5) * 2
            for c in range(3):
                perturbation = noise * sensitivity
                result[:, :, c] += perturbation
                diff = result[:, :, c] - orig[:, :, c]
                diff = np.clip(diff, -eps, eps)
                result[:, :, c] = orig[:, :, c] + diff

        result = np.clip(result, 0, 255).astype(np.uint8)
        output_path = job_dir / "output.png"
        cv2.imwrite(str(output_path), result)

        return FileResponse(
            str(output_path),
            media_type="image/png",
            filename=f"adv_{image.filename.rsplit('.', 1)[0]}.png",
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
        except:
            pass
