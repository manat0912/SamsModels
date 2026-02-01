import colorsys
import gc
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from gradio.themes import Soft
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor, Sam3VideoModel, Sam3VideoProcessor

MODEL_ID = "./models/sam3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

TRACKER_MODEL = Sam3TrackerVideoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()
TRACKER_PROCESSOR = Sam3TrackerVideoProcessor.from_pretrained(MODEL_ID)

TEXT_VIDEO_MODEL = Sam3VideoModel.from_pretrained(MODEL_ID).to(DEVICE, dtype=DTYPE).eval()
TEXT_VIDEO_PROCESSOR = Sam3VideoProcessor.from_pretrained(MODEL_ID)
print("Models loaded successfully!")

MAX_SECONDS = 8.0


def to_device_recursive(obj: Any, device: str | torch.device) -> Any:  # noqa: ANN401
    """Return a new object where all torch.Tensors reachable from `obj` are moved to the given device.

    - Does NOT mutate the original object.
    - Handles:
        * torch.Tensor
        * Mapping (e.g. dict, defaultdict, OrderedDict, etc.)
        * Sequence (e.g. list, tuple) except str/bytes
        * Custom classes with attributes (__dict__)
    - Tries to preserve container types where reasonable.
    """
    device = torch.device(device)
    memo = {}

    def _convert(x: Any) -> Any:  # noqa: ANN401, C901
        obj_id = id(x)
        if obj_id in memo:
            return memo[obj_id]

        # 1. Tensor
        if isinstance(x, torch.Tensor):
            y = x.to(device)
            memo[obj_id] = y
            return y

        # 2. Mapping (dict, defaultdict, etc.)
        if isinstance(x, Mapping):
            # Special case: defaultdict
            if isinstance(x, defaultdict):
                y = defaultdict(x.default_factory)
                memo[obj_id] = y
                for k, v in x.items():
                    y[k] = _convert(v)
                return y

            # Try to rebuild the same type using (key, value) pairs
            try:
                y = type(x)((k, _convert(v)) for k, v in x.items())
                memo[obj_id] = y
                return y
            except TypeError:
                # Fallback: plain dict
                y = {k: _convert(v) for k, v in x.items()}
                memo[obj_id] = y
                return y

        # 3. Sequence (list/tuple/etc.) but not str/bytes
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
            if isinstance(x, list):
                y = [_convert(v) for v in x]
            elif isinstance(x, tuple):
                y = type(x)(_convert(v) for v in x)
            else:
                try:
                    y = type(x)(_convert(v) for v in x)
                except TypeError:
                    y = [_convert(v) for v in x]
            memo[obj_id] = y
            return y

        # 4. Custom object with attributes (__dict__)
        if hasattr(x, "__dict__") and not isinstance(x, type):
            new_obj = x.__class__.__new__(x.__class__)
            memo[obj_id] = new_obj
            for name, value in vars(x).items():
                setattr(new_obj, name, _convert(value))
            return new_obj

        # 5. Everything else → keep as-is
        memo[obj_id] = x
        return x

    return _convert(obj)


def try_load_video_frames(video_path_or_url: str) -> tuple[list[Image.Image], dict]:
    cap = cv2.VideoCapture(video_path_or_url)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    info = {
        "num_frames": len(frames),
        "fps": float(fps_val) if fps_val and fps_val > 0 else None,
    }
    return frames, info


def overlay_masks_on_frame(
    frame: Image.Image,
    masks_per_object: dict[int, np.ndarray],
    color_by_obj: dict[int, tuple[int, int, int]],
    alpha: float = 0.5,
) -> Image.Image:
    base = np.array(frame).astype(np.float32) / 255.0
    height, width = base.shape[:2]
    overlay = base.copy()

    for obj_id, mask in masks_per_object.items():
        if mask is None:
            continue
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        if mask.ndim == 3:
            mask = mask.squeeze()
        mask = np.clip(mask, 0.0, 1.0)
        color = np.array(color_by_obj.get(obj_id, (255, 0, 0)), dtype=np.float32) / 255.0
        a = alpha
        m = mask[..., None]
        overlay = (1.0 - a * m) * overlay + (a * m) * color

    out = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def pastel_color_for_object(obj_id: int) -> tuple[int, int, int]:
    golden_ratio_conjugate = 0.61
    hue = (obj_id * golden_ratio_conjugate) % 1.0
    saturation = 0.45
    value = 1.0
    r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r_f * 255), int(g_f * 255), int(b_f * 255)


def pastel_color_for_prompt(prompt_text: str) -> tuple[int, int, int]:
    """Generate a consistent color for a prompt text using a deterministic hash."""
    # Use a deterministic hash by summing character codes
    # This ensures the same prompt always gets the same color
    char_sum = sum(ord(c) for c in prompt_text)

    # Use the sum to generate a hue that's well-distributed across the color spectrum
    # Multiply by a large prime to spread values out
    hue = ((char_sum * 2654435761) % 360) / 360.0

    # Use pastel colors (lower saturation, high value)
    saturation = 0.5
    value = 0.95
    r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r_f * 255), int(g_f * 255), int(b_f * 255)


class AppState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.video_frames: list[Image.Image] = []
        self.inference_session = None
        self.video_fps: float | None = None
        self.masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
        self.color_by_obj: dict[int, tuple[int, int, int]] = {}
        self.color_by_prompt: dict[str, tuple[int, int, int]] = {}
        self.clicks_by_frame_obj: dict[int, dict[int, list[tuple[int, int, int]]]] = {}
        self.boxes_by_frame_obj: dict[int, dict[int, list[tuple[int, int, int, int]]]] = {}
        self.text_prompts_by_frame_obj: dict[int, dict[int, str]] = {}
        self.composited_frames: dict[int, Image.Image] = {}
        self.current_frame_idx: int = 0
        self.current_obj_id: int = 1
        self.current_label: str = "positive"
        self.current_clear_old: bool = True
        self.current_prompt_type: str = "Points"
        self.pending_box_start: tuple[int, int] | None = None
        self.pending_box_start_frame_idx: int | None = None
        self.pending_box_start_obj_id: int | None = None
        self.active_tab: str = "point_box"

    def __repr__(self) -> str:
        return f"AppState(video_frames={len(self.video_frames)}, video_fps={self.video_fps}, masks_by_frame={len(self.masks_by_frame)}, color_by_obj={len(self.color_by_obj)})"

    @property
    def num_frames(self) -> int:
        return len(self.video_frames)


def init_video_session(
    state: AppState, video: str | dict, active_tab: str = "point_box"
) -> tuple[AppState, int, int, Image.Image, str]:
    state.video_frames = []
    state.masks_by_frame = {}
    state.color_by_obj = {}
    state.color_by_prompt = {}
    state.text_prompts_by_frame_obj = {}
    state.clicks_by_frame_obj = {}
    state.boxes_by_frame_obj = {}
    state.composited_frames = {}
    state.inference_session = None
    state.active_tab = active_tab

    video_path: str | None = None
    if isinstance(video, dict):
        video_path = video.get("name") or video.get("path") or video.get("data")
    elif isinstance(video, str):
        video_path = video
    else:
        video_path = None

    if not video_path:
        raise gr.Error("Invalid video input.")

    frames, info = try_load_video_frames(video_path)
    if len(frames) == 0:
        raise gr.Error("No frames could be loaded from the video.")

    trimmed_note = ""
    fps_in = info.get("fps")
    max_frames_allowed = int(MAX_SECONDS * fps_in) if fps_in else len(frames)
    if len(frames) > max_frames_allowed:
        frames = frames[:max_frames_allowed]
        trimmed_note = f" (trimmed to {int(MAX_SECONDS)}s = {len(frames)} frames)"
        if isinstance(info, dict):
            info["num_frames"] = len(frames)
    state.video_frames = frames
    state.video_fps = float(fps_in) if fps_in else None

    raw_video = [np.array(frame) for frame in frames]

    if active_tab == "text":
        processor = TEXT_VIDEO_PROCESSOR
        state.inference_session = processor.init_video_session(
            video=frames,
            inference_device=DEVICE,
            inference_state_device=DEVICE,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=DTYPE,
        )
    else:
        processor = TRACKER_PROCESSOR
        state.inference_session = processor.init_video_session(
            video=raw_video,
            inference_device=DEVICE,
            inference_state_device=DEVICE,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=DTYPE,
        )

    state.inference_session.inference_device = DEVICE
    state.inference_session.processing_device = DEVICE
    state.inference_session.cache.inference_device = DEVICE

    first_frame = frames[0]
    max_idx = len(frames) - 1
    if active_tab == "text":
        status = (
            f"Loaded {len(frames)} frames @ {state.video_fps or 'unknown'} fps{trimmed_note}. "
            f"Device: {DEVICE}, dtype: bfloat16. Ready for text prompting."
        )
    else:
        status = (
            f"Loaded {len(frames)} frames @ {state.video_fps or 'unknown'} fps{trimmed_note}. "
            f"Device: {DEVICE}, dtype: bfloat16. Video session initialized."
        )
    return state, 0, max_idx, first_frame, status


def compose_frame(state: AppState, frame_idx: int) -> Image.Image:
    if state is None or state.video_frames is None or len(state.video_frames) == 0:
        return None
    frame_idx = int(np.clip(frame_idx, 0, len(state.video_frames) - 1))
    frame = state.video_frames[frame_idx]
    masks = state.masks_by_frame.get(frame_idx, {})
    out_img = frame
    if len(masks) != 0:
        out_img = overlay_masks_on_frame(out_img, masks, state.color_by_obj, alpha=0.65)

    clicks_map = state.clicks_by_frame_obj.get(frame_idx)
    if clicks_map:
        draw = ImageDraw.Draw(out_img)
        cross_half = 6
        for obj_id, pts in clicks_map.items():
            for x, y, lbl in pts:
                color = (0, 255, 0) if int(lbl) == 1 else (255, 0, 0)
                draw.line([(x - cross_half, y), (x + cross_half, y)], fill=color, width=2)
                draw.line([(x, y - cross_half), (x, y + cross_half)], fill=color, width=2)
    if (
        state.pending_box_start is not None
        and state.pending_box_start_frame_idx == frame_idx
        and state.pending_box_start_obj_id is not None
    ):
        draw = ImageDraw.Draw(out_img)
        x, y = state.pending_box_start
        cross_half = 6
        color = state.color_by_obj.get(state.pending_box_start_obj_id, (255, 255, 255))
        draw.line([(x - cross_half, y), (x + cross_half, y)], fill=color, width=2)
        draw.line([(x, y - cross_half), (x, y + cross_half)], fill=color, width=2)
    box_map = state.boxes_by_frame_obj.get(frame_idx)
    if box_map:
        draw = ImageDraw.Draw(out_img)
        for obj_id, boxes in box_map.items():
            color = state.color_by_obj.get(obj_id, (255, 255, 255))
            for x1, y1, x2, y2 in boxes:
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

    text_prompts_by_obj = {}
    for frame_texts in state.text_prompts_by_frame_obj.values():
        for obj_id, text_prompt in frame_texts.items():
            if obj_id not in text_prompts_by_obj:
                text_prompts_by_obj[obj_id] = text_prompt

    if text_prompts_by_obj and len(masks) > 0:
        draw = ImageDraw.Draw(out_img)

        # Calculate scale factor based on image size (reference: 720p height = 720)
        img_width, img_height = out_img.size
        reference_height = 720.0
        scale_factor = img_height / reference_height

        # Scale font size (base size ~13 pixels for default font, scale proportionally)
        base_font_size = 13
        font_size = max(10, int(base_font_size * scale_factor))

        # Try to load a scalable font, fall back to default if not available
        try:
            # Try common system fonts
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "arial.ttf",
            ]
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except OSError:
                    continue
            if font is None:
                # Fallback to default font
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        for obj_id, text_prompt in text_prompts_by_obj.items():
            obj_mask = masks.get(obj_id)
            if obj_mask is not None:
                mask_array = np.array(obj_mask)
                if mask_array.size > 0 and np.any(mask_array):
                    rows = np.any(mask_array, axis=1)
                    cols = np.any(mask_array, axis=0)
                    if np.any(rows) and np.any(cols):
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        label_x = int(x_min)
                        # Scale vertical offset and padding
                        vertical_offset = int(20 * scale_factor)
                        padding = max(2, int(4 * scale_factor))
                        label_y = int(y_min) - vertical_offset
                        label_y = max(int(5 * scale_factor), label_y)

                        obj_color = state.color_by_obj.get(obj_id, (255, 255, 255))

                        # Include object ID in the label
                        label_text = f"{text_prompt} - ID {obj_id}"
                        bbox = draw.textbbox((label_x, label_y), label_text, font=font)
                        draw.rectangle(
                            [(bbox[0] - padding, bbox[1] - padding), (bbox[2] + padding, bbox[3] + padding)],
                            fill=obj_color,
                            outline=None,
                            width=0,
                        )
                        draw.text((label_x, label_y), label_text, fill=(255, 255, 255), font=font)

    state.composited_frames[frame_idx] = out_img
    return out_img


def update_frame_display(state: AppState, frame_idx: int) -> Image.Image:
    if state is None or state.video_frames is None or len(state.video_frames) == 0:
        return None
    frame_idx = int(np.clip(frame_idx, 0, len(state.video_frames) - 1))
    cached = state.composited_frames.get(frame_idx)
    if cached is not None:
        return cached
    return compose_frame(state, frame_idx)


def _get_prompt_for_obj(state: AppState, obj_id: int) -> str | None:
    """Get the prompt text associated with an object ID."""
    # Priority 1: Check text_prompts_by_frame_obj (most reliable)
    for frame_texts in state.text_prompts_by_frame_obj.values():
        if obj_id in frame_texts:
            return frame_texts[obj_id].strip()

    # Priority 2: Check inference session mapping
    if state.inference_session is not None and (
        hasattr(state.inference_session, "obj_id_to_prompt_id")
        and obj_id in state.inference_session.obj_id_to_prompt_id
    ):
        prompt_id = state.inference_session.obj_id_to_prompt_id[obj_id]
        if hasattr(state.inference_session, "prompts") and prompt_id in state.inference_session.prompts:
            return state.inference_session.prompts[prompt_id].strip()

    return None


def _ensure_color_for_obj(state: AppState, obj_id: int) -> None:
    """Assign color to object based on its prompt if available, otherwise use object ID."""
    prompt_text = _get_prompt_for_obj(state, obj_id)

    if prompt_text is not None:
        # Ensure prompt has a color assigned
        if prompt_text not in state.color_by_prompt:
            state.color_by_prompt[prompt_text] = pastel_color_for_prompt(prompt_text)
        # Always update to prompt-based color
        state.color_by_obj[obj_id] = state.color_by_prompt[prompt_text]
    elif obj_id not in state.color_by_obj:
        # Fallback to object ID-based color (for point/box prompting mode)
        state.color_by_obj[obj_id] = pastel_color_for_object(obj_id)


@spaces.GPU
def on_image_click(
    img: Image.Image | np.ndarray,
    state: AppState,
    frame_idx: int,
    obj_id: int,
    label: str,
    clear_old: bool,
    evt: gr.SelectData,
) -> tuple[Image.Image, AppState]:
    if state is None or state.inference_session is None:
        return img

    model = TRACKER_MODEL
    processor = TRACKER_PROCESSOR
    state.inference_session = to_device_recursive(state.inference_session, DEVICE)

    x = y = None
    if evt is not None:
        try:
            if hasattr(evt, "index") and isinstance(evt.index, (list, tuple)) and len(evt.index) == 2:
                x, y = int(evt.index[0]), int(evt.index[1])
            elif hasattr(evt, "value") and isinstance(evt.value, dict) and "x" in evt.value and "y" in evt.value:
                x, y = int(evt.value["x"]), int(evt.value["y"])
        except Exception:
            x = y = None

    if x is None or y is None:
        raise gr.Error("Could not read click coordinates.")

    _ensure_color_for_obj(state, int(obj_id))
    ann_frame_idx = int(frame_idx)
    ann_obj_id = int(obj_id)

    if state.current_prompt_type == "Boxes":
        if state.pending_box_start is None:
            frame_clicks = state.clicks_by_frame_obj.setdefault(ann_frame_idx, {})
            frame_clicks[ann_obj_id] = []
            state.composited_frames.pop(ann_frame_idx, None)
            state.pending_box_start = (int(x), int(y))
            state.pending_box_start_frame_idx = ann_frame_idx
            state.pending_box_start_obj_id = ann_obj_id
            state.composited_frames.pop(ann_frame_idx, None)
            return update_frame_display(state, ann_frame_idx)
        x1, y1 = state.pending_box_start
        x2, y2 = int(x), int(y)
        state.pending_box_start = None
        state.pending_box_start_frame_idx = None
        state.pending_box_start_obj_id = None
        state.composited_frames.pop(ann_frame_idx, None)
        x_min, y_min = min(x1, x2), min(y1, y2)
        x_max, y_max = max(x1, x2), max(y1, y2)

        box = [[[x_min, y_min, x_max, y_max]]]
        processor.add_inputs_to_inference_session(
            inference_session=state.inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_boxes=box,
        )

        frame_boxes = state.boxes_by_frame_obj.setdefault(ann_frame_idx, {})
        obj_boxes = frame_boxes.setdefault(ann_obj_id, [])
        obj_boxes.clear()
        obj_boxes.append((x_min, y_min, x_max, y_max))
        state.composited_frames.pop(ann_frame_idx, None)
    else:
        label_int = 1 if str(label).lower().startswith("pos") else 0

        frame_clicks = state.clicks_by_frame_obj.setdefault(ann_frame_idx, {})
        obj_clicks = frame_clicks.setdefault(ann_obj_id, [])

        if bool(clear_old):
            obj_clicks.clear()
            frame_boxes = state.boxes_by_frame_obj.setdefault(ann_frame_idx, {})
            frame_boxes[ann_obj_id] = []
            if hasattr(state.inference_session, "reset_inference_session"):
                pass

        obj_clicks.append((int(x), int(y), int(label_int)))

        points = [[[[click[0], click[1]] for click in obj_clicks]]]
        labels = [[[click[2] for click in obj_clicks]]]

        processor.add_inputs_to_inference_session(
            inference_session=state.inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=points,
            input_labels=labels,
        )
        state.composited_frames.pop(ann_frame_idx, None)

    with torch.no_grad():
        outputs = model(
            inference_session=state.inference_session,
            frame_idx=ann_frame_idx,
        )

    out_mask_logits = processor.post_process_masks(
        [outputs.pred_masks],
        [[state.inference_session.video_height, state.inference_session.video_width]],
        binarize=False,
    )[0]

    mask_2d = (out_mask_logits[0] > 0.0).cpu().numpy()
    masks_for_frame = state.masks_by_frame.setdefault(ann_frame_idx, {})
    masks_for_frame[ann_obj_id] = mask_2d

    state.composited_frames.pop(ann_frame_idx, None)

    state.inference_session = to_device_recursive(state.inference_session, "cpu")

    return update_frame_display(state, ann_frame_idx), state


@spaces.GPU
def on_text_prompt(
    state: AppState,
    frame_idx: int,
    text_prompt: str,
) -> tuple[Image.Image, str, str, AppState]:
    if state is None or state.inference_session is None:
        return None, "Upload a video and enter text prompt.", "**Active prompts:** None"

    model = TEXT_VIDEO_MODEL
    processor = TEXT_VIDEO_PROCESSOR

    if not text_prompt or not text_prompt.strip():
        active_prompts = _get_active_prompts_display(state)
        return update_frame_display(state, int(frame_idx)), "Please enter a text prompt.", active_prompts, state

    frame_idx = int(np.clip(frame_idx, 0, len(state.video_frames) - 1))

    # Parse comma-separated prompts or single prompt
    prompt_texts = [p.strip() for p in text_prompt.split(",") if p.strip()]
    if not prompt_texts:
        active_prompts = _get_active_prompts_display(state)
        return update_frame_display(state, int(frame_idx)), "Please enter a valid text prompt.", active_prompts, state

    state.inference_session = to_device_recursive(state.inference_session, DEVICE)

    # Add text prompt(s) - supports both single string and list of strings
    state.inference_session = processor.add_text_prompt(
        inference_session=state.inference_session,
        text=prompt_texts,  # Pass as list to add multiple at once
    )

    masks_for_frame = state.masks_by_frame.setdefault(frame_idx, {})
    frame_texts = state.text_prompts_by_frame_obj.setdefault(int(frame_idx), {})

    num_objects = 0
    detected_obj_ids = []
    prompt_to_obj_ids_summary = {}

    with torch.no_grad():
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=state.inference_session,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=1,
        ):
            processed_outputs = processor.postprocess_outputs(
                state.inference_session,
                model_outputs,
            )

            current_frame_idx = model_outputs.frame_idx
            if current_frame_idx == frame_idx:
                object_ids = processed_outputs["object_ids"]
                masks = processed_outputs["masks"]
                scores = processed_outputs["scores"]
                prompt_to_obj_ids = processed_outputs.get("prompt_to_obj_ids", {})

                # Update prompt_to_obj_ids summary for status message
                for prompt, obj_ids in prompt_to_obj_ids.items():
                    if prompt not in prompt_to_obj_ids_summary:
                        prompt_to_obj_ids_summary[prompt] = []
                    prompt_to_obj_ids_summary[prompt].extend(
                        [int(oid) for oid in obj_ids if int(oid) not in prompt_to_obj_ids_summary[prompt]]
                    )

                num_objects = len(object_ids)
                if num_objects > 0:
                    if len(scores) > 0:
                        sorted_indices = torch.argsort(scores, descending=True).cpu().tolist()
                    else:
                        sorted_indices = list(range(num_objects))

                    for mask_idx in sorted_indices:
                        current_obj_id = int(object_ids[mask_idx].item())
                        detected_obj_ids.append(current_obj_id)
                        mask_2d = masks[mask_idx].float().cpu().numpy()
                        if mask_2d.ndim == 3:
                            mask_2d = mask_2d.squeeze()
                        mask_2d = (mask_2d > 0.0).astype(np.float32)
                        masks_for_frame[current_obj_id] = mask_2d

                        # Find which prompt detected this object
                        detected_prompt = None
                        for prompt, obj_ids in prompt_to_obj_ids.items():
                            if current_obj_id in obj_ids:
                                detected_prompt = prompt
                                break

                        # Store prompt and assign color
                        if detected_prompt:
                            frame_texts[current_obj_id] = detected_prompt.strip()
                        _ensure_color_for_obj(state, current_obj_id)

    state.composited_frames.pop(frame_idx, None)

    # Build status message with prompt breakdown
    if detected_obj_ids:
        status_parts = [f"Processed text prompt(s) on frame {frame_idx}. Found {num_objects} object(s):"]
        for prompt, obj_ids in prompt_to_obj_ids_summary.items():
            if obj_ids:
                obj_ids_str = ", ".join(map(str, sorted(obj_ids)))
                status_parts.append(f"  • '{prompt}': {len(obj_ids)} object(s) (IDs: {obj_ids_str})")
        status = "\n".join(status_parts)
    else:
        prompts_str = ", ".join([f"'{p}'" for p in prompt_texts])
        status = f"Processed text prompt(s) {prompts_str} on frame {frame_idx}. No objects detected."

    active_prompts = _get_active_prompts_display(state)

    state.inference_session = to_device_recursive(state.inference_session, "cpu")

    return update_frame_display(state, int(frame_idx)), status, active_prompts, state


def _get_active_prompts_display(state: AppState) -> str:
    """Get a formatted string showing all active prompts in the inference session."""
    if state is None or state.inference_session is None:
        return "**Active prompts:** None"

    if hasattr(state.inference_session, "prompts") and state.inference_session.prompts:
        prompts_list = sorted(set(state.inference_session.prompts.values()))
        if prompts_list:
            prompts_str = ", ".join([f"'{p}'" for p in prompts_list])
            return f"**Active prompts:** {prompts_str}"

    return "**Active prompts:** None"


@spaces.GPU
def propagate_masks(state: AppState) -> Iterator[tuple[AppState, str, dict]]:
    if state is None:
        return state, "Load a video first.", gr.update()

    if state.active_tab != "text" and state.inference_session is None:
        return state, "Load a video first.", gr.update()

    total = max(1, state.num_frames)
    processed = 0

    yield state, f"Propagating masks: {processed}/{total}", gr.update()

    last_frame_idx = 0

    with torch.no_grad():
        if state.active_tab == "text":
            if state.inference_session is None:
                yield state, "Text video model not loaded.", gr.update()
                return

            model = TEXT_VIDEO_MODEL
            processor = TEXT_VIDEO_PROCESSOR

            state.inference_session = to_device_recursive(state.inference_session, DEVICE)

            # Collect all unique prompts from existing frame annotations
            text_prompt_to_obj_ids = {}
            for frame_idx, frame_texts in state.text_prompts_by_frame_obj.items():
                for obj_id, text_prompt in frame_texts.items():
                    if text_prompt not in text_prompt_to_obj_ids:
                        text_prompt_to_obj_ids[text_prompt] = []
                    if obj_id not in text_prompt_to_obj_ids[text_prompt]:
                        text_prompt_to_obj_ids[text_prompt].append(obj_id)

            # Also check if there are prompts already in the inference session
            if hasattr(state.inference_session, "prompts") and state.inference_session.prompts:
                for prompt_text in state.inference_session.prompts.values():
                    if prompt_text not in text_prompt_to_obj_ids:
                        text_prompt_to_obj_ids[prompt_text] = []

            for text_prompt in text_prompt_to_obj_ids:
                text_prompt_to_obj_ids[text_prompt].sort()

            if not text_prompt_to_obj_ids:
                state.inference_session = to_device_recursive(state.inference_session, "cpu")
                yield state, "No text prompts found. Please add a text prompt first.", gr.update()
                return

            # Add all prompts to the inference session (processor handles deduplication)
            for text_prompt in text_prompt_to_obj_ids:
                state.inference_session = processor.add_text_prompt(
                    inference_session=state.inference_session,
                    text=text_prompt,
                )

            earliest_frame = min(state.text_prompts_by_frame_obj.keys()) if state.text_prompts_by_frame_obj else 0

            frames_to_track = state.num_frames - earliest_frame

            outputs_per_frame = {}

            for model_outputs in model.propagate_in_video_iterator(
                inference_session=state.inference_session,
                start_frame_idx=earliest_frame,
                max_frame_num_to_track=frames_to_track,
            ):
                processed_outputs = processor.postprocess_outputs(
                    state.inference_session,
                    model_outputs,
                )
                frame_idx = model_outputs.frame_idx
                outputs_per_frame[frame_idx] = processed_outputs

                object_ids = processed_outputs["object_ids"]
                masks = processed_outputs["masks"]
                scores = processed_outputs["scores"]
                prompt_to_obj_ids = processed_outputs.get("prompt_to_obj_ids", {})

                masks_for_frame = state.masks_by_frame.setdefault(frame_idx, {})
                frame_texts = state.text_prompts_by_frame_obj.setdefault(frame_idx, {})

                num_objects = len(object_ids)
                if num_objects > 0:
                    if len(scores) > 0:
                        sorted_indices = torch.argsort(scores, descending=True).cpu().tolist()
                    else:
                        sorted_indices = list(range(num_objects))

                    for mask_idx in sorted_indices:
                        current_obj_id = int(object_ids[mask_idx].item())
                        mask_2d = masks[mask_idx].float().cpu().numpy()
                        if mask_2d.ndim == 3:
                            mask_2d = mask_2d.squeeze()
                        mask_2d = (mask_2d > 0.0).astype(np.float32)
                        masks_for_frame[current_obj_id] = mask_2d

                        # Find which prompt detected this object
                        found_prompt = None
                        for prompt, obj_ids in prompt_to_obj_ids.items():
                            if current_obj_id in obj_ids:
                                found_prompt = prompt
                                break

                        # Store prompt and assign color
                        if found_prompt:
                            frame_texts[current_obj_id] = found_prompt.strip()
                        _ensure_color_for_obj(state, current_obj_id)

                state.composited_frames.pop(frame_idx, None)
                last_frame_idx = frame_idx
                processed += 1
                if processed % 30 == 0 or processed == total:
                    state.inference_session = to_device_recursive(state.inference_session, "cpu")
                    yield state, f"Propagating masks: {processed}/{total}", gr.update(value=frame_idx)
                    state.inference_session = to_device_recursive(state.inference_session, DEVICE)
        else:
            if state.inference_session is None:
                yield state, "Tracker model not loaded.", gr.update()
                return

            model = TRACKER_MODEL
            processor = TRACKER_PROCESSOR

            state.inference_session = to_device_recursive(state.inference_session, DEVICE)

            for sam2_video_output in model.propagate_in_video_iterator(inference_session=state.inference_session):
                video_res_masks = processor.post_process_masks(
                    [sam2_video_output.pred_masks],
                    original_sizes=[[state.inference_session.video_height, state.inference_session.video_width]],
                )[0]

                frame_idx = sam2_video_output.frame_idx
                for i, out_obj_id in enumerate(state.inference_session.obj_ids):
                    _ensure_color_for_obj(state, int(out_obj_id))
                    mask_2d = video_res_masks[i].cpu().numpy()
                    masks_for_frame = state.masks_by_frame.setdefault(frame_idx, {})
                    masks_for_frame[int(out_obj_id)] = mask_2d
                    state.composited_frames.pop(frame_idx, None)

                last_frame_idx = frame_idx
                processed += 1
                if processed % 30 == 0 or processed == total:
                    state.inference_session = to_device_recursive(state.inference_session, "cpu")
                    yield state, f"Propagating masks: {processed}/{total}", gr.update(value=frame_idx)
                    state.inference_session = to_device_recursive(state.inference_session, DEVICE)

    text = f"Propagated masks across {processed} frames."
    state.inference_session = to_device_recursive(state.inference_session, "cpu")
    yield state, text, gr.update(value=last_frame_idx)


def reset_prompts(state: AppState) -> tuple[AppState, Image.Image, str, str]:
    """Reset prompts and all outputs, but keep processed frames and cached vision features."""
    if state is None or state.inference_session is None:
        active_prompts = _get_active_prompts_display(state)
        return state, None, "No active session to reset.", active_prompts

    if state.active_tab != "text":
        active_prompts = _get_active_prompts_display(state)
        return state, None, "Reset prompts is only available for text prompting mode.", active_prompts

    # Reset inference session tracking data but keep cache and processed frames
    if hasattr(state.inference_session, "reset_tracking_data"):
        state.inference_session.reset_tracking_data()

    # Manually clear prompts (reset_tracking_data doesn't clear prompts themselves)
    if hasattr(state.inference_session, "prompts"):
        state.inference_session.prompts.clear()
    if hasattr(state.inference_session, "prompt_input_ids"):
        state.inference_session.prompt_input_ids.clear()
    if hasattr(state.inference_session, "prompt_embeddings"):
        state.inference_session.prompt_embeddings.clear()
    if hasattr(state.inference_session, "prompt_attention_masks"):
        state.inference_session.prompt_attention_masks.clear()
    if hasattr(state.inference_session, "obj_id_to_prompt_id"):
        state.inference_session.obj_id_to_prompt_id.clear()

    # Reset detection-tracking fusion state
    if hasattr(state.inference_session, "obj_id_to_score"):
        state.inference_session.obj_id_to_score.clear()
    if hasattr(state.inference_session, "obj_id_to_tracker_score_frame_wise"):
        state.inference_session.obj_id_to_tracker_score_frame_wise.clear()
    if hasattr(state.inference_session, "obj_id_to_last_occluded"):
        state.inference_session.obj_id_to_last_occluded.clear()
    if hasattr(state.inference_session, "max_obj_id"):
        state.inference_session.max_obj_id = -1
    if hasattr(state.inference_session, "obj_first_frame_idx"):
        state.inference_session.obj_first_frame_idx.clear()
    if hasattr(state.inference_session, "unmatched_frame_inds"):
        state.inference_session.unmatched_frame_inds.clear()
    if hasattr(state.inference_session, "overlap_pair_to_frame_inds"):
        state.inference_session.overlap_pair_to_frame_inds.clear()
    if hasattr(state.inference_session, "trk_keep_alive"):
        state.inference_session.trk_keep_alive.clear()
    if hasattr(state.inference_session, "removed_obj_ids"):
        state.inference_session.removed_obj_ids.clear()
    if hasattr(state.inference_session, "suppressed_obj_ids"):
        state.inference_session.suppressed_obj_ids.clear()
    if hasattr(state.inference_session, "hotstart_removed_obj_ids"):
        state.inference_session.hotstart_removed_obj_ids.clear()

    # Clear all app state outputs
    state.masks_by_frame.clear()
    state.text_prompts_by_frame_obj.clear()
    state.composited_frames.clear()
    state.color_by_obj.clear()
    state.color_by_prompt.clear()

    # Update display
    current_idx = int(getattr(state, "current_frame_idx", 0))
    current_idx = max(0, min(current_idx, state.num_frames - 1))
    preview_img = update_frame_display(state, current_idx)
    active_prompts = _get_active_prompts_display(state)
    status = "Prompts and outputs reset. Processed frames and cached vision features preserved."

    return state, preview_img, status, active_prompts


def reset_session(state: AppState) -> tuple[AppState, Image.Image, int, int, str, str]:
    if not state.video_frames:
        return state, None, 0, 0, "Session reset. Load a new video.", "**Active prompts:** None"

    if state.active_tab == "text":
        if state.video_frames:
            processor = TEXT_VIDEO_PROCESSOR
            state.inference_session = processor.init_video_session(
                video=state.video_frames,
                inference_device=DEVICE,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=DTYPE,
            )
    elif state.inference_session is not None and hasattr(state.inference_session, "reset_inference_session"):
        state.inference_session.reset_inference_session()
    elif state.video_frames:
        processor = TRACKER_PROCESSOR
        raw_video = [np.array(frame) for frame in state.video_frames]
        state.inference_session = processor.init_video_session(
            video=raw_video,
            inference_device=DEVICE,
            video_storage_device="cpu",
            processing_device="cpu",
            dtype=DTYPE,
        )

    state.masks_by_frame.clear()
    state.clicks_by_frame_obj.clear()
    state.boxes_by_frame_obj.clear()
    state.text_prompts_by_frame_obj.clear()
    state.composited_frames.clear()
    state.color_by_obj.clear()
    state.color_by_prompt.clear()
    state.pending_box_start = None
    state.pending_box_start_frame_idx = None
    state.pending_box_start_obj_id = None

    gc.collect()

    current_idx = int(getattr(state, "current_frame_idx", 0))
    current_idx = max(0, min(current_idx, state.num_frames - 1))
    preview_img = update_frame_display(state, current_idx)
    slider_minmax = gr.update(minimum=0, maximum=max(state.num_frames - 1, 0), interactive=True)
    slider_value = gr.update(value=current_idx)
    status = "Session reset. Prompts cleared; video preserved."
    active_prompts = _get_active_prompts_display(state)
    return state, preview_img, slider_minmax, slider_value, status, active_prompts


def _on_video_change_pointbox(state: AppState, video: str | dict) -> tuple[AppState, dict, Image.Image, str]:
    state, min_idx, max_idx, first_frame, status = init_video_session(state, video, "point_box")
    return (
        state,
        gr.update(minimum=min_idx, maximum=max_idx, value=min_idx, interactive=True),
        first_frame,
        status,
    )


def _on_video_change_text(state: AppState, video: str | dict) -> tuple[AppState, dict, Image.Image, str, str]:
    if video is None:
        return state, None, None, None, None
    state, min_idx, max_idx, first_frame, status = init_video_session(state, video, "text")
    active_prompts = _get_active_prompts_display(state)
    return (
        state,
        gr.update(minimum=min_idx, maximum=max_idx, value=min_idx, interactive=True),
        first_frame,
        status,
        active_prompts,
    )


with gr.Blocks(title="SAM3", theme=Soft(primary_hue="blue", secondary_hue="rose", neutral_hue="slate")) as demo:
    app_state = gr.State(AppState())

    gr.Markdown(
        """
        ### SAM3 Video Tracking · powered by Hugging Face 🤗 Transformers
        Segment and track objects across a video with SAM3 (Segment Anything 3). This demo runs the official implementation from the Hugging Face Transformers library for interactive, promptable video segmentation with point, box, and text prompts.
        """
    )

    with gr.Tabs() as main_tabs:
        with gr.Tab("Text Prompting"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Quick start**
                        - **Load a video**: Upload your own or pick an example below.
                        - Select a frame and enter text description(s) to segment objects (e.g., "red car", "penguin"). You can add multiple prompts separated by commas (e.g., "person, bed, lamp") or add them one by one. The text prompt will return all the instances of the object in the frame and not specific ones (e.g. not "penguin on the left" but "penguin").
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **Working with results**
                        - **Preview**: Use the slider to navigate frames and see the current masks.
                        - **Propagate**: Click "Propagate across video" to track all defined objects through the entire video.
                        - **Export**: Render an MP4 for smooth playback using the original video FPS.
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    video_in_text = gr.Video(label="Upload video", sources=["upload", "webcam"])
                    load_status_text = gr.Markdown(visible=True)
                    reset_btn_text = gr.Button("Reset Session", variant="secondary")
                with gr.Column(scale=2):
                    preview_text = gr.Image(label="Preview")
                    with gr.Row():
                        frame_slider_text = gr.Slider(label="Frame", minimum=0, maximum=0, step=1, value=0)
                        with gr.Column(scale=0):
                            propagate_btn_text = gr.Button("Propagate across video", variant="primary")
                            propagate_status_text = gr.Markdown(visible=True)
                    with gr.Row():
                        text_prompt_input = gr.Textbox(
                            label="Text Prompt(s)",
                            placeholder="Enter text description(s) (e.g., 'person' or 'person, bed, lamp' for multiple)",
                            lines=2,
                        )
                        with gr.Column(scale=0):
                            text_apply_btn = gr.Button("Apply Text Prompt(s)", variant="primary")
                            reset_prompts_btn = gr.Button("Reset Prompts", variant="secondary")
                    active_prompts_display = gr.Markdown("**Active prompts:** None", visible=True)
                    text_status = gr.Markdown(visible=True)

            with gr.Row():
                render_btn_text = gr.Button("Render MP4 for smooth playback", variant="primary")
            playback_video_text = gr.Video(label="Rendered Playback", interactive=False)

            examples_list_text = [
                [None, "./deers.mp4"],
                [None, "./penguins.mp4"],
                [None, "./foot.mp4"],
            ]
            with gr.Row():
                gr.Examples(
                    label="Examples",
                    examples=examples_list_text,
                    inputs=[app_state, video_in_text],
                    examples_per_page=5,
                )

        with gr.Tab("Point/Box Prompting"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **Quick start**
                        - **Load a video**: Upload your own or pick an example below.
                        - Select an Object ID and point label (positive/negative), then click the frame to add guidance. You can add **multiple points per object** and define **multiple objects** across frames.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **Working with results**
                        - **Preview**: Use the slider to navigate frames and see the current masks.
                        - **Propagate**: Click "Propagate across video" to track all defined objects through the entire video.
                        - **Export**: Render an MP4 for smooth playback using the original video FPS.
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    video_in_pointbox = gr.Video(label="Upload video", sources=["upload", "webcam"])
                    load_status_pointbox = gr.Markdown(visible=True)
                    reset_btn_pointbox = gr.Button("Reset Session", variant="secondary")
                with gr.Column(scale=2):
                    preview_pointbox = gr.Image(label="Preview")
                    with gr.Row():
                        frame_slider_pointbox = gr.Slider(label="Frame", minimum=0, maximum=0, step=1, value=0)
                        with gr.Column(scale=0):
                            propagate_btn_pointbox = gr.Button("Propagate across video", variant="primary")
                            propagate_status_pointbox = gr.Markdown(visible=True)

            with gr.Row():
                obj_id_inp = gr.Number(value=1, precision=0, label="Object ID", scale=0)
                label_radio = gr.Radio(choices=["positive", "negative"], value="positive", label="Point label")
                clear_old_chk = gr.Checkbox(value=False, label="Clear old inputs for this object")
                prompt_type = gr.Radio(choices=["Points", "Boxes"], value="Points", label="Prompt type")

            with gr.Row():
                render_btn_pointbox = gr.Button("Render MP4 for smooth playback", variant="primary")
            playback_video_pointbox = gr.Video(label="Rendered Playback", interactive=False)

            examples_list_pointbox = [
                [None, "./deers.mp4"],
                [None, "./penguins.mp4"],
                [None, "./foot.mp4"],
            ]
            with gr.Row():
                gr.Examples(
                    label="Examples",
                    examples=examples_list_pointbox,
                    inputs=[app_state, video_in_pointbox],
                    examples_per_page=5,
                )

    video_in_pointbox.change(
        fn=_on_video_change_pointbox,
        inputs=[app_state, video_in_pointbox],
        outputs=[app_state, frame_slider_pointbox, preview_pointbox, load_status_pointbox],
        show_progress=True,
    )

    def _sync_frame_idx_pointbox(state_in: AppState, idx: int) -> Image.Image:
        if state_in is not None:
            state_in.current_frame_idx = int(idx)
        return update_frame_display(state_in, int(idx))

    frame_slider_pointbox.change(
        fn=_sync_frame_idx_pointbox,
        inputs=[app_state, frame_slider_pointbox],
        outputs=preview_pointbox,
    )

    video_in_text.change(
        fn=_on_video_change_text,
        inputs=[app_state, video_in_text],
        outputs=[app_state, frame_slider_text, preview_text, load_status_text, active_prompts_display],
        show_progress=True,
    )

    def _sync_frame_idx_text(state_in: AppState, idx: int) -> Image.Image:
        if state_in is not None:
            state_in.current_frame_idx = int(idx)
        return update_frame_display(state_in, int(idx))

    frame_slider_text.change(
        fn=_sync_frame_idx_text,
        inputs=[app_state, frame_slider_text],
        outputs=preview_text,
    )

    def _sync_obj_id(s: AppState, oid: int) -> None:
        if s is not None and oid is not None:
            s.current_obj_id = int(oid)

    obj_id_inp.change(
        fn=_sync_obj_id,
        inputs=[app_state, obj_id_inp],
    )

    def _sync_label(s: AppState, lab: str) -> None:
        if s is not None and lab is not None:
            s.current_label = str(lab)

    label_radio.change(
        fn=_sync_label,
        inputs=[app_state, label_radio],
    )

    def _sync_prompt_type(s: AppState, val: str) -> tuple[dict, dict]:
        if s is not None and val is not None:
            s.current_prompt_type = str(val)
            s.pending_box_start = None
        is_points = str(val).lower() == "points"
        return (
            gr.update(visible=is_points),
            gr.update(interactive=is_points) if is_points else gr.update(value=True, interactive=False),
        )

    prompt_type.change(
        fn=_sync_prompt_type,
        inputs=[app_state, prompt_type],
        outputs=[label_radio, clear_old_chk],
    )

    preview_pointbox.select(
        fn=on_image_click,
        inputs=[preview_pointbox, app_state, frame_slider_pointbox, obj_id_inp, label_radio, clear_old_chk],
        outputs=[preview_pointbox, app_state],
    )

    text_apply_btn.click(
        fn=on_text_prompt,
        inputs=[app_state, frame_slider_text, text_prompt_input],
        outputs=[preview_text, text_status, active_prompts_display, app_state],
    )

    reset_prompts_btn.click(
        fn=reset_prompts,
        inputs=app_state,
        outputs=[app_state, preview_text, text_status, active_prompts_display],
    )

    def _render_video(s: AppState) -> str:
        if s is None or s.num_frames == 0:
            raise gr.Error("Load a video first.")
        fps = s.video_fps if s.video_fps and s.video_fps > 0 else 12
        frames_np = []
        first = compose_frame(s, 0)
        h, w = first.size[1], first.size[0]
        for idx in range(s.num_frames):
            img = s.composited_frames.get(idx)
            if img is None:
                img = compose_frame(s, idx)
            frames_np.append(np.array(img)[:, :, ::-1])
            if (idx + 1) % 60 == 0:
                gc.collect()
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path.name, fourcc, fps, (w, h))
                for fr_bgr in frames_np:
                    writer.write(fr_bgr)
                writer.release()
                return out_path.name
        except Exception as e:
            print(f"Failed to render video with cv2: {e}")
            raise gr.Error(f"Failed to render video: {e}")

    render_btn_pointbox.click(
        fn=_render_video,
        inputs=app_state,
        outputs=playback_video_pointbox,
    )
    render_btn_text.click(
        fn=_render_video,
        inputs=app_state,
        outputs=playback_video_text,
    )

    propagate_btn_pointbox.click(
        fn=propagate_masks,
        inputs=app_state,
        outputs=[app_state, propagate_status_pointbox, frame_slider_pointbox],
    )

    propagate_btn_text.click(
        fn=propagate_masks,
        inputs=app_state,
        outputs=[app_state, propagate_status_text, frame_slider_text],
    )

    reset_btn_pointbox.click(
        fn=reset_session,
        inputs=app_state,
        outputs=[app_state, preview_pointbox, frame_slider_pointbox, frame_slider_pointbox, load_status_pointbox],
    )

    reset_btn_text.click(
        fn=reset_session,
        inputs=app_state,
        outputs=[
            app_state,
            preview_text,
            frame_slider_text,
            frame_slider_text,
            load_status_text,
            active_prompts_display,
        ],
    )


demo.queue(api_open=False).launch()
