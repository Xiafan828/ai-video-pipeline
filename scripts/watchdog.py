# -*- coding: utf-8 -*-
"""看门狗：等主流程生成出成片后自动关机

监控逻辑：
  1. 每 60 秒检查一次：
     - gen.log 是否还在更新（主流程还活着）
     - shinian_final.mp4 是否已生成
  2. 成片出现 → 等 60 秒确保写完 → 关机（60 秒倒计时，可 shutdown /a 取消）
  3. 主流程超过 90 分钟无任何日志更新 → 认为卡死，记录后不关机（保留现场供排查）
"""
import os
import subprocess
import time
from datetime import datetime

WORK = r"E:\ds工作区\临时\shinian_build"
LOG = os.path.join(WORK, "gen.log")
FINAL = os.path.join(WORK, "shinian_final.mp4")
WDLOG = os.path.join(WORK, "watchdog.log")

MAX_SILENCE_MIN = 90      # 日志超过这么久没更新 => 判定卡死
CHECK_INTERVAL = 60


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(WDLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log("看门狗启动，监控 %s" % FINAL)
    while True:
        time.sleep(CHECK_INTERVAL)

        # 成片出现（且大小稳定）
        if os.path.exists(FINAL) and os.path.getsize(FINAL) > 100_000:
            s1 = os.path.getsize(FINAL)
            time.sleep(20)
            s2 = os.path.getsize(FINAL)
            if s1 == s2:
                log("成片已稳定: %.1f MB" % (s2 / 1e6))
                # 记录成片信息
                try:
                    r = subprocess.run(
                        [r"E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\static_ffmpeg.exe",
                         "-i", FINAL],
                        capture_output=True, text=True, encoding="utf-8", errors="ignore")
                    for line in r.stderr.split("\n"):
                        if "Duration" in line or "Stream #0:" in line:
                            log("  " + line.strip()[:90])
                except Exception as e:
                    log("  读取信息失败: %s" % str(e)[:60])
                log("准备关机（60 秒倒计时，可 shutdown /a 取消）")
                subprocess.run(["shutdown", "/s", "/t", "60", "/c", "H3 十年 数字人视频已生成完毕"],
                               capture_output=True)
                return

        # 卡死检测
        if os.path.exists(LOG):
            age = (time.time() - os.path.getmtime(LOG)) / 60
            if age > MAX_SILENCE_MIN:
                log("警告：gen.log 已 %.0f 分钟未更新，疑似卡死，不关机，保留现场" % age)
                return
        else:
            log("gen.log 还不存在，主流程可能未启动")


if __name__ == "__main__":
    main()
