from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "https://vip.DMXapi.com/v1"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_API_KEY_ENV = "DMX_API_KEY"

DEFAULT_PROMPT_TEXT = """You are given an image showing a sequence of robot actions arranged from top to bottom. Analyze the whole action in order, from top to bottom, and describe what the robot is doing in detailed, natural English.
Use vivid, precise language to describe the movements, positions, and interactions of the robot's parts (e.g., hands, arms, torso). You should detect the key semantics of the robot action, and write one descriptive sentence for all the actions.
Examples of the desired style:
'The palm is thrust towards the companion, as if pushing something into his face.'
'The hands move up and down alternately, with the palms brushing against one another as they pass.'
'The right hand is placed on the abdomen, slightly bend down towards the target to be thanked, and the left hand is pulled back to the left rear.'
'Lift the forearm to chest height, parallel to the ground, and quickly cross it from the opposite side to the side of the corresponding arm.'
'The pursed hand is jerked towards the open mouth several times.'
'The shoulders are hunched up briefly and the hands are offered in a palm-up position with the fingers spread.'
'The hand is raised with the thumb, forefinger and little finger all spread. The other two fingers are bent down.'
Output the descriptions in order, separated by new lines."""


def _encode_image_b64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_mime_type(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    return mime or "application/octet-stream"


def _extract_text_from_response(data: dict[str, Any]) -> str:
    """
    兼容 OpenAI 兼容接口的返回结构：
      - choices[0].message.content
    """
    try:
        choices = data.get("choices") or []
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
            return "\n".join([x.strip() for x in parts if x.strip()]).strip()
    except Exception:
        pass
    return ""


def describe_image(
    image_path: Path,
    *,
    prompt_text: str = DEFAULT_PROMPT_TEXT,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    temperature: float = 0.1,
    request_user: str = "DMXAPI",
    timeout_sec: float = 120.0,
    max_retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    key = api_key or os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"未找到 API Key。请设置环境变量 {api_key_env}，或通过 --api-key 传入。")

    api_url = base_url.rstrip("/") + "/chat/completions"
    mime = _guess_mime_type(image_path)
    b64 = _encode_image_b64(image_path)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
        "temperature": float(temperature),
        "user": request_user,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "User-Agent": f"motion-diffusion-model/1.0 ({base_url})",
    }

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            return {
                "ok": True,
                "model": model,
                "base_url": base_url,
                "image_path": str(image_path),
                "mime_type": mime,
                "prompt": prompt_text,
                "response_json": data,
                "text": _extract_text_from_response(data),
            }
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            time.sleep(retry_backoff_sec * (2**attempt))

    raise RuntimeError(f"调用多模态接口失败: {last_err}") from last_err


def main() -> None:
    ap = argparse.ArgumentParser(description="调用多模态大模型，为图片生成动作描述（输出 JSON + 可选纯文本）")
    ap.add_argument("--image", type=Path, required=True, help="输入图片路径（例如 concat_vertical.jpg）")
    ap.add_argument("--out-json", type=Path, default=None, help="输出 JSON 路径（默认 <image>.desc.json）")
    ap.add_argument("--out-txt", type=Path, default=None, help="输出纯文本路径（默认 <image>.desc.txt）")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", type=str, default=None, help="直接传 API Key（不建议，优先用环境变量）")
    ap.add_argument("--api-key-env", type=str, default=DEFAULT_API_KEY_ENV, help="从该环境变量读取 API Key")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--prompt-path", type=Path, default=None, help="提示词文本文件路径（可选）")
    ap.add_argument("--timeout-sec", type=float, default=120.0)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    prompt_text = DEFAULT_PROMPT_TEXT
    if args.prompt_path is not None:
        prompt_text = args.prompt_path.read_text(encoding="utf-8")

    out_json = args.out_json or args.image.with_suffix(args.image.suffix + ".desc.json")
    out_txt = args.out_txt or args.image.with_suffix(args.image.suffix + ".desc.txt")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    result = describe_image(
        args.image,
        prompt_text=prompt_text,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        timeout_sec=args.timeout_sec,
        max_retries=int(args.max_retries),
    )

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text((result.get("text") or "").strip() + "\n", encoding="utf-8")
    print(f"[OK] json: {out_json}")
    print(f"[OK] txt : {out_txt}")


if __name__ == "__main__":
    main()
