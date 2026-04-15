import os
import uuid
import shutil
import subprocess
import numpy as np
import io
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WORK_DIR = Path("/tmp/metaclean")
WORK_DIR.mkdir(exist_ok=True)

def cleanup_stale_jobs(max_age_seconds: int = 600):
    """Remove job dirs mais velhos que max_age_seconds para liberar disco."""
    import time
    now = time.time()
    for d in WORK_DIR.iterdir():
        if d.is_dir():
            try:
                age = now - d.stat().st_mtime
                if age > max_age_seconds:
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

# ═══════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    cleanup_stale_jobs()  # limpa jobs velhos a cada health check (a cada 10s)
    return {"status": "ok"}

# ═══════════════════════════════════════════════════════════════════════════
# TTS — White Script (apenas geração de voz)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/tts")
async def tts_endpoint(
    text: str = Form(...),
    lang: str = Form("pt"),
):
    try:
        from gtts import gTTS
        lang_code = lang.split("-")[0].lower()
        buf = io.BytesIO()
        gTTS(text=text, lang=lang_code, slow=False).write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg",
                                  headers={"Content-Disposition": "inline; filename=tts.mp3"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ═══════════════════════════════════════════════════════════════════════════
# PROCESS AUDIO — White Script com STFT (substituição espectral + silence gate)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/process/audio")
async def process_audio_ws(
    audio: UploadFile = File(...),
    text: str = Form(...),
    lang: str = Form("pt"),
    mix_db: float = Form(-20.0),
):
    import scipy.signal
    import scipy.io.wavfile
    from gtts import gTTS

    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        # ── Salvar áudio original ──
        input_path = job_dir / f"input{_ext(audio.filename)}"
        with open(input_path, "wb") as f:
            f.write(await audio.read())

        # ── Converter para WAV 44100Hz stereo via ffmpeg ──
        wav_path = job_dir / "converted.wav"
        run_ffmpeg(["ffmpeg", "-y", "-i", str(input_path),
                    "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", str(wav_path)])

        sr, data = scipy.io.wavfile.read(str(wav_path))  # int16

        # ── Gerar TTS ──
        lang_code = lang.split("-")[0].lower()
        tts_buf = io.BytesIO()
        gTTS(text=text, lang=lang_code, slow=False).write_to_fp(tts_buf)
        tts_buf.seek(0)
        tts_mp3 = job_dir / "tts.mp3"
        tts_wav  = job_dir / "tts.wav"
        with open(tts_mp3, "wb") as f:
            f.write(tts_buf.read())
        run_ffmpeg(["ffmpeg", "-y", "-i", str(tts_mp3),
                    "-ar", str(sr), "-ac", "1", "-sample_fmt", "s16", str(tts_wav)])

        _, tts_data = scipy.io.wavfile.read(str(tts_wav))  # int16 mono

        # ── Processar: substituição espectral STFT + silence gate ──
        output = inject_white_script_stft(data, tts_data, sr, mix_db)

        # ── Salvar WAV de saída ──
        output_path = job_dir / "output.wav"
        scipy.io.wavfile.write(str(output_path), sr, output)

        fname = audio.filename.rsplit(".", 1)[0]
        return FileResponse(str(output_path), media_type="audio/wav",
                            filename=f"ws_{fname}.wav",
                            background=BackgroundTask(shutil.rmtree, job_dir, True))

    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return JSONResponse({"error": str(e)}, status_code=500)


def inject_white_script_stft(original: np.ndarray, tts: np.ndarray, sr: int, mix_db: float) -> np.ndarray:
    """
    Injeção subliminar de TTS na banda de voz (300–3400 Hz).
    Estratégia: ADIÇÃO em volume baixo por baixo do original (não substituição).
    O original é preservado integralmente — humano não percebe o TTS.
    STT detecta porque analisa energia espectral matematicamente.
    mix_db controla o volume do TTS relativo ao original:
      -20 dB = muito sutil (quase inaudível, recomendado)
      -12 dB = levemente audível
    """
    import scipy.signal

    # Normalizar para float32 [-1, 1]
    orig_f = original.astype(np.float32) / 32768.0
    tts_f  = tts.astype(np.float32) / 32768.0

    # Separar canais (stereo)
    if orig_f.ndim == 2:
        channels = [orig_f[:, i] for i in range(orig_f.shape[1])]
    else:
        channels = [orig_f]

    # Loop TTS para cobrir toda a duração
    n = len(channels[0])
    if len(tts_f) < n:
        reps = int(np.ceil(n / len(tts_f)))
        tts_f = np.tile(tts_f, reps)
    tts_f = tts_f[:n]

    # STFT params
    nperseg  = 1024
    noverlap = nperseg * 3 // 4
    freq_res = sr / nperseg
    low_bin  = max(1, int(300  / freq_res))
    high_bin = min(nperseg // 2, int(3400 / freq_res))

    # Fator de redução do original na banda de voz
    # -15dB → original fica a 40% (TTS a 60%)
    # -28dB → original fica a 20% (TTS a 80%)
    # -40dB → original fica a 10% (TTS a 90%)
    t = (-mix_db - 15) / 25.0          # 0.0 em -15dB, 1.0 em -40dB
    orig_duck = np.clip(0.40 - t * 0.30, 0.10, 0.40)  # de 0.40 até 0.10

    # STFT do TTS
    _, _, tts_stft = scipy.signal.stft(tts_f, fs=sr, nperseg=nperseg, noverlap=noverlap)

    output_channels = []
    for ch in channels:
        _, _, orig_stft = scipy.signal.stft(ch, fs=sr, nperseg=nperseg, noverlap=noverlap)

        # RMS por frame na banda de voz
        orig_band_rms = np.sqrt(np.mean(np.abs(orig_stft[low_bin:high_bin, :]) ** 2, axis=0) + 1e-10)
        tts_band_rms  = np.sqrt(np.mean(np.abs(tts_stft[low_bin:high_bin, :]) ** 2, axis=0) + 1e-10)

        # Threshold silêncio
        silence_thr = np.percentile(orig_band_rms, 35)

        out_stft = orig_stft.copy()
        n_frames = orig_stft.shape[1]

        for i in range(n_frames):
            ti = i % tts_stft.shape[1]

            # Normalizar TTS para igualar RMS do original frame-a-frame
            norm = orig_band_rms[i] / (tts_band_rms[ti] + 1e-10)
            norm = np.clip(norm, 0.1, 10.0)
            tts_norm = tts_stft[:, ti] * norm

            is_silence = orig_band_rms[i] <= silence_thr

            if is_silence:
                # Silêncio: TTS em volume pleno — STT capta claramente
                out_stft[low_bin:high_bin, i] = tts_norm[low_bin:high_bin]
            else:
                # Fala ativa: abafa o original na banda de voz + adiciona TTS normalizado
                # Humano ouve o original mais baixo (parece estilo de mixagem)
                # STT capta o TTS que está no mesmo nível do original abafado
                out_stft[low_bin:high_bin, i] = (
                    orig_stft[low_bin:high_bin, i] * orig_duck +
                    tts_norm[low_bin:high_bin] * (1.0 - orig_duck)
                )
            # Fora da banda de voz: original intacto (graves e agudos normais)

        _, out_ch = scipy.signal.istft(out_stft, fs=sr, nperseg=nperseg, noverlap=noverlap)
        out_ch = np.clip(out_ch[:n], -1.0, 1.0)
        output_channels.append(out_ch)

    # Recombinar canais
    out = np.stack(output_channels, axis=1) if len(output_channels) > 1 else output_channels[0]
    return (out * 32767).astype(np.int16)

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

    info = get_video_info(input_path)
    fps = info["fps"]
    width = info["width"]
    height = info["height"]
    duration = info["duration"]

    speed_factor = 0.99 + np.random.random() * 0.02

    extract_frames(input_path, frames_dir, fps)

    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".png")],
        key=lambda x: int(x.split(".")[0].split("_")[-1]) if "_" in x else 0,
    )

    if not frame_files:
        raise ValueError("No frames extracted from video")

    if cover_path and os.path.exists(cover_path):
        cover_img = cv2.imread(cover_path)
        cover_img = cv2.resize(cover_img, (width, height))
    else:
        cover_img = cv2.imread(os.path.join(frames_dir, frame_files[0]))

    intro_count = max(1, int(fps * 0.5))
    frame_idx = 0
    for i in range(intro_count):
        intro = generate_intro_frame(width, height)
        cv2.imwrite(os.path.join(processed_dir, f"frame_{frame_idx:06d}.png"), intro)
        frame_idx += 1

    cover_processed = cover_img.copy()
    cover_processed = apply_crop_and_scale(cover_processed, width, height)
    cover_processed = apply_color_shift(cover_processed)
    cover_processed = apply_noise(cover_processed, seed=42, eps=4, tiktok=tiktok, frame_idx=0)
    cv2.imwrite(os.path.join(processed_dir, f"frame_{frame_idx:06d}.png"), cover_processed)
    frame_idx += 1

    for i, fname in enumerate(frame_files):
        frame = cv2.imread(os.path.join(frames_dir, fname))
        if frame is None:
            continue

        frame = apply_crop_and_scale(frame, width, height)
        frame = apply_color_shift(frame)

        if level >= 2:
            frame = apply_overlay(frame, cover_img, alpha=0.06)

        seed = 1000 + i
        frame = apply_noise(frame, seed=seed, eps=4, tiktok=tiktok, frame_idx=i)

        if i % 3 == 0:
            frame = apply_invisible_pixels(frame, seed=seed)

        if level == 3 and i % 15 == 0:
            frame = apply_color_flash(frame)

        cv2.imwrite(os.path.join(processed_dir, f"frame_{frame_idx:06d}.png"), frame)
        frame_idx += 1

    for i in range(intro_count):
        outro = generate_intro_frame(width, height)
        cv2.imwrite(os.path.join(processed_dir, f"frame_{frame_idx:06d}.png"), outro)
        frame_idx += 1

    target_duration = {1: 0, 2: 300, 3: 600}.get(level, 0)
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

    effective_fps = fps * speed_factor
    temp_video = os.path.join(job_dir, "temp_video.mp4")

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

    if audio_path and os.path.exists(audio_path):
        merge_audio(temp_video, audio_path, output_path, target_duration if level >= 2 else duration)
    else:
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
    crop_pct = 0.01 + np.random.random() * 0.02
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
    hue_shift = (np.random.random() - 0.5) * 10
    sat_mult = 0.97 + np.random.random() * 0.06
    val_mult = 0.97 + np.random.random() * 0.06
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
    import cv2
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
    r1, g1, b1 = np.random.randint(0, 40, 3)
    r2, g2, b2 = np.random.randint(0, 60, 3)
    for y in range(height):
        t = y / height
        frame[y, :, 0] = int(b1 * (1 - t) + b2 * t)
        frame[y, :, 1] = int(g1 * (1 - t) + g2 * t)
        frame[y, :, 2] = int(r1 * (1 - t) + r2 * t)
    noise = np.random.randint(-4, 5, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# FFMPEG HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_video_info(path):
    import json
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not video_stream:
        raise ValueError("No video stream found")

    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
    fps = min(fps, 60)

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
        pass


def merge_audio(video_path, audio_path, output_path, target_duration):
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
# IMAGE ATTACK
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

        img = apply_crop_and_scale(img, w, h)
        img = apply_micro_rotation(img)
        img = apply_color_shift(img)

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
