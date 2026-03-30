from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # 允许直接 `python api/xxx.py` 运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.extract_and_concat_frames import extract_frames_and_concat_vertical  # noqa: E402
from api.image_understand import DEFAULT_MODEL, DEFAULT_PROMPT_TEXT, describe_image  # noqa: E402


def _iter_video_files(root: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def _safe_stem(p: Path) -> str:
    # 用文件名做文件夹名即可（motion_stat_300/videos 下是唯一的）
    return p.stem


def process_one_video(
    video_path: Path,
    *,
    out_root: Path,
    interval_sec: float,
    duration_sec: float,
    prompt_text: str,
    model: str,
    base_url: str,
    api_key: str | None,
    api_key_env: str,
    temperature: float,
    skip_existing: bool,
    no_understand: bool,
) -> dict[str, Any]:
    vid_dir = out_root / _safe_stem(video_path)
    frames_dir = vid_dir / "frames_concat"
    frames_dir.mkdir(parents=True, exist_ok=True)

    concat_img = frames_dir / "concat_vertical.jpg"
    desc_json = vid_dir / "description.json"
    desc_txt = vid_dir / "description.txt"
    meta_json = vid_dir / "meta.json"

    if skip_existing and concat_img.exists() and (no_understand or (desc_json.exists() and desc_txt.exists())):
        return {"video": str(video_path), "skipped": True, "out_dir": str(vid_dir)}

    # 1) 抽帧 + 竖向拼接图（会覆盖旧文件）
    concat_img = extract_frames_and_concat_vertical(
        video_path=video_path,
        output_dir=frames_dir,
        interval_sec=interval_sec,
        duration_sec=duration_sec,
    )

    text_out = ""
    if not no_understand:
        # 2) 多模态理解 + 落盘
        result = describe_image(
            concat_img,
            prompt_text=prompt_text,
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            temperature=temperature,
        )
        text_out = (result.get("text") or "").strip()
        desc_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        desc_txt.write_text(text_out + "\n", encoding="utf-8")

    meta = {
        "video_path": str(video_path),
        "out_dir": str(vid_dir),
        "concat_image": str(concat_img),
        "interval_sec": interval_sec,
        "duration_sec": duration_sec,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "temperature": temperature,
        "no_understand": no_understand,
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"video": str(video_path), "skipped": False, "out_dir": str(vid_dir), "text": text_out}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="批量：对 data/motion_stat_300/videos 下视频渲染长图，并调用多模态生成文本描述（保存图片+文本，便于人工检查）"
    )
    ap.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/motion_stat_300/videos"),
        help="视频目录（递归遍历）",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/motion_stat_300_understand"),
        help="输出根目录",
    )
    ap.add_argument("--interval", type=float, default=0.5, help="截帧间隔(秒)")
    ap.add_argument("--duration", type=float, default=5.0, help="截帧总时长(秒)")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（0 表示全部）")
    ap.add_argument("--skip-existing", action="store_true", help="若输出已存在则跳过")
    ap.add_argument("--no-understand", action="store_true", help="只渲染图片，不调用大模型")

    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--base-url", type=str, default=os.environ.get("DMX_BASE_URL", "https://vip.DMXapi.com/v1"))
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--api-key-env", type=str, default="DMX_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--prompt-path", type=Path, default=None, help="自定义提示词文件（可选）")
    args = ap.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval 必须大于 0")
    if args.duration <= 0:
        raise ValueError("--duration 必须大于 0")
    if not args.videos_dir.exists():
        raise FileNotFoundError(f"videos-dir 不存在: {args.videos_dir}")

    prompt_text = DEFAULT_PROMPT_TEXT
    if args.prompt_path is not None:
        prompt_text = args.prompt_path.read_text(encoding="utf-8")

    args.out_root.mkdir(parents=True, exist_ok=True)

    videos = _iter_video_files(args.videos_dir)
    if args.limit and args.limit > 0:
        videos = videos[: int(args.limit)]

    if not videos:
        raise FileNotFoundError(f"未找到视频文件: {args.videos_dir}")

    summary_path = args.out_root / "summary.jsonl"
    processed = 0
    skipped = 0
    failed = 0
    with open(summary_path, "a", encoding="utf-8") as sf:
        for vp in videos:
            try:
                row = process_one_video(
                    vp,
                    out_root=args.out_root,
                    interval_sec=float(args.interval),
                    duration_sec=float(args.duration),
                    prompt_text=prompt_text,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    api_key_env=args.api_key_env,
                    temperature=float(args.temperature),
                    skip_existing=bool(args.skip_existing),
                    no_understand=bool(args.no_understand),
                )
            except Exception as e:
                failed += 1
                row = {
                    "video": str(vp),
                    "skipped": False,
                    "failed": True,
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=20),
                }
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed += 1
            if row.get("skipped"):
                skipped += 1

    print("[Done]")
    print(f"  out_root: {args.out_root}")
    print(f"  summary : {summary_path}")
    print(f"  videos  : {processed} (skipped {skipped}, failed {failed})")


if __name__ == "__main__":
    main()

