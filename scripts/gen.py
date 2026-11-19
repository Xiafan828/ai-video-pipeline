# -*- coding: utf-8 -*-
"""《十年》逐段生成 + 验证 + 拼接

每段流程：
  1. 写 first_frame 图片（第1段用 FINAL_FACE.png，之后用上一段末帧）
  2. 提交 ComfyUI（chain_id 每段唯一，seed_policy=fixed 便于复现）
  3. 等完成 → 找到输出的 mp4
  4. whisper 验证：
     - 唱段：转写文本和台词做字符相似度（>= 0.6 通过，因为唱出来会走音）
     - 间奏段：转写结果应该很短或空
  5. 不合格 → 换 seed 重试（最多 MAX_RETRY 次）
  6. 合格 → 抽末帧（第 frames-1 帧）作为下一段 first_frame
全部完成后 → ffmpeg 顺序拼接
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import wave

COMFY = "http://127.0.0.1:8188"
COMFY_ROOT = r"E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI"
FF = os.path.join(COMFY_ROOT, ".venv", "Scripts", "static_ffmpeg.exe")
WORK = r"E:\ds工作区\临时\shinian_build"
INPUT_DIR = os.path.join(COMFY_ROOT, "input")
OUT_ROOT = os.path.join(COMFY_ROOT, "output", "H3_shinian")
PLAN = os.path.join(WORK, "plan.json")
SHOTS = os.path.join(WORK, "shots")
FACE = "FINAL_FACE.png"

FPS = 24
W, H = 512, 768
STEPS = 20
MAX_RETRY = 3
LOG = os.path.join(WORK, "gen.log")

os.makedirs(SHOTS, exist_ok=True)


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fmt_t(t):
    m = int(t // 60)
    s = t - m * 60
    return "%02d:%06.3f" % (m, s)


def build_api(seg, idx, img_name, chain_id, seed):
    dur = seg["frames"] / FPS
    if seg["type"] == "sing":
        line = "，".join(seg["texts"])
        timeline = (
            "During 00:00.000–%s, she sings this line once, exactly these characters, "
            "in a natural singing voice following a steady melodic phrase, completing it before the interval ends. "
            "This is the ONLY text she sings in this interval. She never adds other words, "
            "never speaks English, never improvises. <d>[Mandarin]%s</d>"
            % (fmt_t(dur), line)
        )
        glob = ("subject_definitions:\n"
                "<Subject 1> is the woman in <Picture 1>, same face, hairstyle and clothing throughout.\n"
                "summary:\n"
                "One continuous close-up portrait shot, fixed camera, soft indoor lighting. She sings directly to camera.\n"
                "overall_soundscape:\n"
                "Only her voice; no backing music.")
    else:
        timeline = (
            "During 00:00.000–%s, she is resting quietly between phrases. "
            "She holds the same position facing the camera, "
            "her mouth stays slightly open as she breathes calmly, "
            "she keeps eye contact with the camera, a soft warm smile showing a hint of teeth, "
            "sways her upper body gently from side to side at a calm even tempo, "
            "blinks naturally two or three times, tilts her head slightly to one side and back, "
            "her shoulders rising and falling with calm deep breaths. "
            "The room is quiet and still around her."
            % fmt_t(dur)
        )
        glob = ("subject_definitions:\n"
                "<Subject 1> is the woman in <Picture 1>, same face, hairstyle and clothing throughout.\n"
                "summary:\n"
                "One continuous close-up portrait shot, fixed camera, soft indoor lighting. She rests quietly between phrases.\n"
                "overall_soundscape:\n"
                "Quiet room tone and calm soft breathing.")

    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "5": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "6": {"class_type": "MiniMaxH3PromptRelayPlanT8Advanced", "inputs": {
            "global_prompt": glob,
            "local_prompts": timeline,
            "length": seg["frames"],
            "timing_mode": "auto_equal",
            "time_ranges": "0-%d" % (seg["frames"] - 1),
            "math_profile": "paper_v1",
            "epsilon": 0.1,
            "allow_gaps": False,
            "allow_overlaps": False}},
        "8": {"class_type": "MiniMaxH3PromptRelayQueryRouteT8Advanced", "inputs": {
            "query_route": "joint_av_exp", "prompt_relay_plan": ["6", 0]}},
        "7": {"class_type": "MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced", "inputs": {
            "model": ["11", 0], "clip": ["2", 0], "video_vae": ["3", 0], "audio_vae": ["4", 0],
            "prompt_relay_plan": ["8", 0], "first_frame": ["5", 0],
            "chain_id": chain_id,
            "total_duration_seconds": dur,
            "width": W, "height": H,
            "render_window_frames": seg["frames"],
            "context_frames": 39,
            "global_prompt": "", "segment_prompts_json": "",
            "prompt_relay_mode": "apply_exp", "query_chunk_rows": 256,
            "eav_mode": "disabled", "eav_tau": 4.0,
            "eav_start_video_progress": 0.15, "eav_end_video_progress": 0.9,
            "eav_max_workspace_mib": 32, "eav_g_hard_limit": 1.5,
            "minimum_free_vram_mib": 512,
            "base_seed": seed, "seed_policy": "fixed",
            "steps": STEPS, "shift_video": 12.0, "shift_audio": 3.0,
            "sampler_name": "dual_clock_euler", "scheduler": "native_flow",
            "task_type": "auto",
            "context_audio": "video_and_audio", "audio_mode": "native",
            "audio_denoise_strength": 0.35,
            "prompt_primary_audio_ordinal": 0, "strict_prompt_tags": True,
            "ref_image_size": "match", "reference_video_policy": "official_2_to_15s",
            "first_frame_reuse": "segment0_only",
            "add_source_as_reference": True,
            "persistent_identity_strategy": "single_reference",
            "persistent_identity_interval": 1, "resume_existing": False,
            "filename_prefix": "H3_shinian/s%03d" % idx,
            "audio_seam_policy": "cosine_bridge", "bridge_ms": 5.0,
            "bit_depth": 8, "crf": 18, "model_id": "unknown"}},
        "10": {"class_type": "MiniMaxH3LowVRAMAttentionT8Advanced", "inputs": {"head_chunks": 4, "model": ["1", 0]}},
        "11": {"class_type": "MiniMaxH3ChunkFeedForwardT8Advanced", "inputs": {"chunks": 2, "seq_threshold": 4096, "model": ["10", 0]}},
    }


def submit(api):
    req = urllib.request.Request(COMFY + "/prompt", data=json.dumps({"prompt": api}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        return r.get("prompt_id"), None
    except Exception as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode("utf-8", errors="ignore")[:400]
            except Exception:
                pass
        return None, body or str(e)[:200]


def wait(pid, timeout=2700):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(20)
        try:
            h = json.loads(urllib.request.urlopen(COMFY + "/history/" + pid, timeout=30).read())
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    for m in st.get("messages", []):
                        if m[0] == "execution_error":
                            return False, "节点%s: %s" % (m[1].get("node_type"), str(m[1].get("exception_message"))[:250])
                    return False, "执行失败"
                return True, h[pid].get("outputs", {})
        except Exception:
            pass
    return False, "超时"


def find_video(outputs):
    """从 outputs 里找 mp4 文件名"""
    for k, v in outputs.items():
        if isinstance(v, dict) and "images" in v:
            for item in v["images"]:
                fn = item.get("filename", "")
                if fn.endswith(".mp4"):
                    return os.path.join(COMFY_ROOT, "output", item.get("subfolder", ""), fn)
    return None


# ============ whisper 验证 ============
_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(mp4):
    """抽音频 + whisper 转写"""
    wav = os.path.join(WORK, "_tmp.wav")
    subprocess.run([FF, "-y", "-i", mp4, "-ac", "1", "-ar", "16000", "-vn", wav],
                   capture_output=True)
    if not os.path.exists(wav):
        return ""
    with wave.open(wav, "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < 16000:
        return ""
    m = get_whisper()
    segs, _ = m.transcribe(audio, language="zh", beam_size=5, condition_on_previous_text=False)
    return "".join(s.text for s in segs).strip()


def clean(s):
    return re.sub(r'[，。！？、；：""''（）\s·]', '', s)


def similarity(a, b):
    """拼音级相似度：容忍 whisper 的繁简差异和同音错字"""
    import difflib
    try:
        from pypinyin import lazy_pinyin
        pa = "".join(lazy_pinyin(clean(a)))
        pb = "".join(lazy_pinyin(clean(b)))
        return difflib.SequenceMatcher(None, pa, pb).ratio()
    except Exception:
        return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()


# ============ 主流程 ============
def make_static_shot(frame_png, frames, out_mp4):
    """兜底：用静帧 + 轻微平移缩放合成一段（H3 间奏念水印时使用）"""
    dur = frames / FPS
    vf = ("zoompan=z='1.02+0.012*sin(2*PI*on/168)'"
          ":x='iw/2-(iw/zoom/2)+7*sin(2*PI*on/96)'"
          ":y='ih/2-(ih/zoom/2)'"
          ":d=1:s=%dx%d:fps=%d,format=yuv420p" % (W, H, FPS))
    cmd = [FF, "-y", "-loop", "1", "-i", frame_png,
           "-f", "lavfi", "-i", "anullsrc=r=32000:cl=stereo",
           "-t", "%.3f" % dur, "-vf", vf,
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "128k", "-shortest", out_mp4]
    subprocess.run(cmd, capture_output=True)
    return os.path.exists(out_mp4)


def main():
    plan = json.load(open(PLAN, encoding="utf-8"))
    state_path = os.path.join(WORK, "state.json")
    state = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else {"done": {}, "prev_frame": None}

    log("=== 开始，共 %d 段 ===" % len(plan))

    for i, seg in enumerate(plan):
        key = str(i)
        if key in state["done"] and os.path.exists(state["done"][key]):
            log("段%02d 已完成，跳过" % i)
            continue

        # 准备 first_frame
        if i == 0:
            img_name = FACE
        else:
            prev = state.get("prev_frame")
            if not prev or not os.path.exists(prev):
                log("段%02d 缺上一段末帧，中止" % i)
                return
            img_name = "shinian_f%03d.png" % i
            shutil.copy2(prev, os.path.join(INPUT_DIR, img_name))

        ok = False
        mp4 = None          # 每次换段必须重置，否则降级会误用上一段的视频
        for attempt in range(MAX_RETRY):
            seed = 20260918 + i * 100 + attempt * 7
            chain = "shinian_%03d_%d" % (i, attempt)
            log("段%02d %s %d帧 第%d次 seed=%d" % (i, seg["type"], seg["frames"], attempt + 1, seed))
            api = build_api(seg, i, img_name, chain, seed)
            pid, err = submit(api)
            if not pid:
                log("  提交失败: %s" % err)
                continue
            log("  prompt=%s 等待…" % pid)
            ok2, out = wait(pid)
            if not ok2:
                log("  执行失败: %s" % out)
                continue
            mp4 = find_video(out if isinstance(out, dict) else {})
            if not mp4 or not os.path.exists(mp4):
                log("  找不到输出文件")
                continue

            txt = transcribe(mp4)
            if seg["type"] == "sing":
                target = "，".join(seg["texts"])
                sim = similarity(txt, target)
                log("  转写: %s" % txt[:60])
                log("  目标: %s" % target[:60])
                log("  相似度: %.3f" % sim)
                if sim >= 0.85:
                    ok = True
            else:
                n = len(clean(txt))
                # H3 训练数据污染：会念出视频水印（"字幕by索兰娅"之类）
                WATERMARK = ("字幕", "索兰娅", "翻译", "校对", "制作", "上传",
                             "订阅", "频道", "音乐", "分享", "原创", "版")
                has_wm = any(w in txt for w in WATERMARK) or "by" in txt.lower()
                only_hum = re.fullmatch(r'[哼啊嗯哈呃哦噢呜唉呀嘛]+', clean(txt)) is not None
                log("  间奏转写: %d 字 【%s】 水印=%s 纯气声=%s" % (n, txt[:40], has_wm, only_hum))
                if n <= 6 and not has_wm and (only_hum or n <= 2):
                    ok = True

            if ok:
                dst = os.path.join(SHOTS, "shot_%03d.mp4" % i)
                shutil.copy2(mp4, dst)
                state["done"][key] = dst
                # 抽末帧
                last = os.path.join(SHOTS, "shot_%03d_last.png" % i)
                subprocess.run([FF, "-y", "-sseof", "-0.2", "-i", dst, "-update", "1",
                                "-q:v", "2", last], capture_output=True)
                if os.path.exists(last):
                    state["prev_frame"] = last
                log("  ✓ 通过，末帧: %s" % (last if os.path.exists(last) else "失败"))
                json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                break
            else:
                log("  ✗ 不合格，重试")

        if not ok:
            # 间奏段念水印时，用静帧兜底（保证零语音）
            if seg["type"] == "interlude":
                src = os.path.join(INPUT_DIR, img_name) if i > 0 else os.path.join(INPUT_DIR, FACE)
                dst = os.path.join(SHOTS, "shot_%03d_STATIC.mp4" % i)
                log("段%02d 间奏重试失败，改用静帧兜底" % i)
                if make_static_shot(src, seg["frames"], dst):
                    state["done"][key] = dst
                    state.setdefault("static", []).append(i)
                    last = os.path.join(SHOTS, "shot_%03d_last.png" % i)
                    subprocess.run([FF, "-y", "-sseof", "-0.2", "-i", dst, "-update", "1",
                                    "-q:v", "2", last], capture_output=True)
                    if os.path.exists(last):
                        state["prev_frame"] = last
                    json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                    log("  静帧兜底完成: %s" % dst)
                    continue
            # 其他情况：降级使用最后一版
            log("段%02d 重试 %d 次仍失败，降级使用最后一版并继续" % (i, MAX_RETRY))
            if mp4 and os.path.exists(mp4):
                dst = os.path.join(SHOTS, "shot_%03d_BAD.mp4" % i)
                shutil.copy2(mp4, dst)
                state["done"][key] = dst
                state.setdefault("bad", []).append(i)
                last = os.path.join(SHOTS, "shot_%03d_last.png" % i)
                subprocess.run([FF, "-y", "-sseof", "-0.2", "-i", dst, "-update", "1",
                                "-q:v", "2", last], capture_output=True)
                if os.path.exists(last):
                    state["prev_frame"] = last
                json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            else:
                log("段%02d 无可用输出，中止" % i)
                return

    # 拼接
    log("=== 全部完成，开始拼接 ===")
    files = [state["done"][str(i)] for i in range(len(plan)) if str(i) in state["done"]]
    lst = os.path.join(WORK, "concat.txt")
    with open(lst, "w", encoding="utf-8", newline="\n") as f:
        for p in files:
            f.write("file '%s'\n" % p.replace("\\", "/"))
    final = os.path.join(WORK, "shinian_final.mp4")
    subprocess.run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", final],
                   capture_output=True)
    log("成片: %s" % final)
    if os.path.exists(final):
        info = subprocess.run([FF, "-i", final], capture_output=True, text=True, encoding="utf-8", errors="ignore").stderr
        for line in info.split("\n"):
            if "Duration" in line:
                log("  " + line.strip()[:60])


if __name__ == "__main__":
    main()
