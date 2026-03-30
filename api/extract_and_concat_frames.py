import argparse
from pathlib import Path
import subprocess


def ensure_output_dir(video_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    default_dir = video_path.parent / f"{video_path.stem}_frames_concat"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir


def run_cmd(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}\n{exc.stderr}") from exc


def extract_frames_and_concat_vertical(
    video_path: Path,
    output_dir: Path | None,
    interval_sec: float = 0.5,
    duration_sec: float = 5.0,
) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    save_dir = ensure_output_dir(video_path, output_dir)
    frame_count = int(duration_sec / interval_sec)
    frame_pattern = save_dir / "frame_%02d.jpg"

    # 先在0~duration范围内按interval抽帧，并限制总帧数。
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-t",
            str(duration_sec),
            "-vf",
            f"fps=1/{interval_sec}",
            "-frames:v",
            str(frame_count),
            str(frame_pattern),
        ]
    )

    frame_files = [save_dir / f"frame_{i:02d}.jpg" for i in range(1, frame_count + 1)]
    missing = [str(p) for p in frame_files if not p.exists()]
    if missing:
        raise RuntimeError(f"部分帧未生成: {missing}")

    inputs: list[str] = []
    for frame in frame_files:
        inputs.extend(["-i", str(frame)])

    long_image_path = save_dir / "concat_vertical.jpg"
    run_cmd(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            f"vstack=inputs={frame_count}",
            str(long_image_path),
        ]
    )
    return long_image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="每0.5秒截帧并上下拼接成长图")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(
            "/root/workspace/motion-diffusion-model/semi_videos/sem5s/"
            "000000_1_wayne_0_100_100__07_iconic_h__0000__11.869_with_wav.mp4"
        ),
        help="输入视频路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，不传则自动在视频同级新建文件夹",
    )
    parser.add_argument("--interval", type=float, default=0.5, help="截帧间隔(秒)")
    parser.add_argument("--duration", type=float, default=5.0, help="截帧总时长(秒)")
    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval 必须大于 0")
    if args.duration <= 0:
        raise ValueError("--duration 必须大于 0")

    result = extract_frames_and_concat_vertical(
        video_path=args.video,
        output_dir=args.output_dir,
        interval_sec=args.interval,
        duration_sec=args.duration,
    )
    print(f"完成，拼接长图已保存: {result}")


if __name__ == "__main__":
    main()
