# -*- coding: utf-8 -*-
"""按 LRC 歌词边界，把一首歌切成「约10.8秒」的段
   规则：每段从一句的起点开始，到下几句累积到最接近 260 帧（10.83s）的那句结束
   帧数对齐 H3 的 17n+5 网格
"""
import re
import sys

FPS = 24
GRID = 17


def grid_frames(seconds, prefer="ceil"):
    """把时长换算成 17n+5 网格上的帧数"""
    want = seconds * FPS
    n = max(1, round((want - 5) / GRID))
    up = GRID * n + 5
    while up < want:
        n += 1
        up = GRID * n + 5
    if prefer == "ceil":
        return up
    down = up - GRID
    return down if down >= 5 else up


def parse_lrc(path):
    """返回 [(起始秒, 歌词)]"""
    out = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r'\[(\d+):(\d+)[.:](\d+)\](.*)', line)
        if not m:
            continue
        t = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100.0
        txt = m.group(4).strip()
        if not txt:
            continue
        if any(k in txt for k in ("作曲", "编曲", "制作人", "录音", "混音", "作词", "合声")):
            continue
        out.append((t, txt))
    return out


def plan_segments(lrc, target_sec=10.83, tail_min=3.0):
    """贪心切分：每段累积到最接近 target_sec 的句子边界"""
    lines = list(lrc)
    # 每句的结束时间 = 下一句的起点（最后一句用 +6 秒估算）
    spans = []
    for i, (t, txt) in enumerate(lines):
        end = lines[i + 1][0] if i + 1 < len(lines) else t + 6.0
        spans.append((t, end, txt))

    segs = []
    i = 0
    while i < len(spans):
        start = spans[i][0]
        best_j = i
        best_err = None
        j = i
        while j < len(spans):
            dur = spans[j][1] - start
            err = abs(dur - target_sec)
            # 允许略微超过 target，但不允许远少于
            if dur > target_sec * 1.35 and best_err is not None:
                break
            if best_err is None or err < best_err:
                best_err = err
                best_j = j
            j += 1
        end = spans[best_j][1]
        segs.append({
            "start": start,
            "end": end,
            "dur": end - start,
            "frames": grid_frames(end - start),
            "lines": [spans[k][2] for k in range(i, best_j + 1)],
        })
        i = best_j + 1
    return segs


if __name__ == "__main__":
    lrc_path = sys.argv[1] if len(sys.argv) > 1 else r"E:\ds工作区\临时\lyric_test\shinian.lrc"
    lrc = parse_lrc(lrc_path)
    print("歌词 %d 句" % len(lrc))
    segs = plan_segments(lrc)
    print("切成 %d 段\n" % len(segs))
    print("%-4s %-8s %-8s %-7s %-7s %s" % ("段", "起点", "终点", "时长", "帧数", "歌词"))
    print("=" * 92)
    total = 0
    for k, s in enumerate(segs, 1):
        total += s["frames"]
        txt = " / ".join(s["lines"])
        print("%-4d %8.2f %8.2f %6.2fs %6d  %s" % (k, s["start"], s["end"], s["dur"], s["frames"], txt[:52]))
        if len(txt) > 52:
            print("%38s %s" % ("", txt[52:100]))
    print("=" * 92)
    print("总帧数: %d  (%.2f 秒)" % (total, total / FPS))
