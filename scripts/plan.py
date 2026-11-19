# -*- coding: utf-8 -*-
"""《十年》切段（最终版）

规则：
  1. 用 LRC 起点切句
  2. 用 whisper 块间隔 >5s 定性为「静默区」，切成 interlude 段
     - 包括开头前奏、中间间奏、结尾尾奏
  3. 每段 <= 260 帧（17n+5 网格）
"""
import json
import os

import numpy as np

FPS = 24
MAX_FRAMES = 260
GRID = 17
WORK = r"E:\ds工作区\临时\shinian_build"
LYRIC = r"E:\ds工作区\临时\lyric_test"
AUDIO_END = 205.42
os.makedirs(WORK, exist_ok=True)


def snap_frames(seconds):
    """吸附到 17n+5 网格，且不低于 H3 最小值 124 帧"""
    want = seconds * FPS
    n = max(1, int(np.ceil((want - 5) / GRID)))
    f = GRID * n + 5
    return max(124, min(f, MAX_FRAMES))


def build():
    d = json.load(open(os.path.join(LYRIC, "lines.json"), encoding="utf-8"))
    lrc = d["lrc"]
    blocks = d["blocks"]

    # 1) 歌曲时间轴拆成「唱」和「静默」两大类
    events = []

    # 开头前奏
    if blocks and blocks[0]["end"] - 0 > 5.0:
        events.append(("interlude", 0.0, blocks[0]["end"]))
    # 中间静默
    for i in range(len(blocks) - 1):
        g = blocks[i + 1]["start"] - blocks[i]["end"]
        if g > 5.0:
            events.append(("interlude", round(blocks[i]["end"], 2), round(blocks[i + 1]["start"], 2)))
    # 结尾尾奏
    if blocks and AUDIO_END - blocks[-1]["end"] > 5.0:
        events.append(("interlude", round(blocks[-1]["end"], 2), AUDIO_END))

    # 2) 把「唱」的区间填进去：每句 = LRC 起点 → 下一句起点
    sing_events = []
    for i, L in enumerate(lrc):
        nxt = lrc[i + 1]["start"] if i + 1 < len(lrc) else AUDIO_END
        sing_events.append(("sing", L["start"], nxt, [L["text"]]))

    # 3) 合并成一条时间轴：按起点排序，静默覆盖掉重叠的唱段
    timeline = []
    for e in sing_events:
        typ, a, b, txts = e
        # 被静默区覆盖的部分剔除
        pieces = [(a, b)]
        for (_, ga, gb) in events:
            newp = []
            for (pa, pb) in pieces:
                if pb <= ga or pa >= gb:
                    newp.append((pa, pb))
                else:
                    if pa < ga:
                        newp.append((pa, ga))
                    if pb > gb:
                        newp.append((gb, pb))
            pieces = newp
        for (pa, pb) in pieces:
            if pb - pa > 1.5:   # 小于1.5秒的残片丢弃（属于上一句的尾巴）
                timeline.append({"type": "sing", "start": round(pa, 2), "end": round(pb, 2), "texts": list(txts)})
    for (_, ga, gb) in events:
        timeline.append({"type": "interlude", "start": round(ga, 2), "end": round(gb, 2), "texts": []})
    timeline.sort(key=lambda x: x["start"])

    # 4) 合并同类（不跨静默边界），每段 <= MAX_FRAMES
    merged = []
    for it in timeline:
        if not merged:
            merged.append(dict(it))
            continue
        last = merged[-1]
        if it["type"] == last["type"] and (it["end"] - last["start"]) <= MAX_FRAMES / FPS:
            last["end"] = it["end"]
            last["texts"] += it["texts"]
        else:
            merged.append(dict(it))

    # 5) 超长硬切
    final = []
    for s in merged:
        dur = s["end"] - s["start"]
        limit = MAX_FRAMES / FPS
        if dur <= limit + 0.2:
            final.append(dict(s))
        else:
            n = int(np.ceil(dur / limit))
            step = dur / n
            for k in range(n):
                a = s["start"] + k * step
                b = min(s["end"], a + step)
                final.append({"type": s["type"], "start": round(a, 2), "end": round(b, 2),
                              "texts": list(s["texts"]) if k == 0 else []})

    # 6) 吸附帧数
    for s in final:
        s["frames"] = snap_frames(s["end"] - s["start"])
        s["dur"] = round(s["frames"] / FPS, 3)
    return final


if __name__ == "__main__":
    segs = build()
    total = sum(s["frames"] for s in segs)
    n_sing = sum(1 for s in segs if s["type"] == "sing")
    n_int = len(segs) - n_sing
    print("共 %d 段（唱 %d / 间奏 %d），总 %d 帧 = %.1f 秒\n" % (len(segs), n_sing, n_int, total, total / FPS))
    print("%-4s %-10s %-8s %-8s %-6s %s" % ("段", "类型", "起点", "终点", "帧", "内容"))
    print("=" * 100)
    for i, s in enumerate(segs):
        txt = " / ".join(s["texts"]) if s["texts"] else "(间奏·静默)"
        print("%-4d %-10s %7.2f %8.2f %6d  %s" % (i, s["type"], s["start"], s["end"], s["frames"], txt[:50]))
        if len(txt) > 50:
            print("%42s %s" % ("", txt[50:96]))
    print("=" * 100)
    print("预计生成时长：%d 段 × 平均 %.0f 分钟 ≈ %.1f 小时（一次通过）" % (
        len(segs), 16, len(segs) * 16 / 60))
    json.dump(segs, open(os.path.join(WORK, "plan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("已保存 plan.json")
