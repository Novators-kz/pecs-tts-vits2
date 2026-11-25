#!/usr/bin/env python3
"""Create and validate mixed static INT8 candidates from VITS2 ONNX models."""

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


MODEL_NAMES = ("vits2_eng_girl.onnx", "vits2_kaz_girl.onnx", "vits2_rus_girl.onnx", "vits2_kaz_nao.onnx", "vits2_rus_nao.onnx")
SCALES = np.array([0.667, 1.0, 0.8], dtype=np.float32)
REPOSITORY_DIR = Path(__file__).resolve().parent.parent  # pecs-tts-vits2 root
FP32_DIR = REPOSITORY_DIR  # fp32 models sit in the repo root (git-ignored)


class SyntheticTokenReader(CalibrationDataReader):
    """Data-free calibration with valid model token IDs and realistic sequence lengths."""

    def __init__(self, vocab_size: int, sample_count: int):
        # shortcut: token IDs are synthetic; replace with production phonemes when tts_inference is restored.
        rng = np.random.default_rng(42)
        self.samples = []
        for length in rng.integers(12, 121, size=sample_count):
            tokens = rng.integers(1, vocab_size, size=(1, int(length)), dtype=np.int64)
            self.samples.append({
                "input": tokens,
                "input_lengths": np.array([tokens.shape[1]], dtype=np.int64),
                "scales": SCALES,
            })

    def get_next(self):
        return self.samples.pop(0) if self.samples else None


def vocab_size(model_path: Path) -> int:
    model = onnx.load(model_path, load_external_data=False)
    return next(initializer.dims[0] for initializer in model.graph.initializer if initializer.name == "sid")


def decoder_conv_nodes(model_path: Path) -> list[str]:
    model = onnx.load(model_path, load_external_data=False)
    return [
        node.name
        for node in model.graph.node
        if node.op_type == "Conv" and node.name.startswith("/dec/")
    ]


def compare_audio(source: Path, candidate: Path, vocab: int) -> tuple[float, float]:
    tokens = (np.arange(48, dtype=np.int64) % (vocab - 1) + 1)[None, :]
    feed = {
        "input": tokens,
        "input_lengths": np.array([tokens.shape[1]], dtype=np.int64),
        "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    fp32 = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feed)[0].reshape(-1)
    int8 = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"]).run(None, feed)[0].reshape(-1)
    if fp32.shape != int8.shape or not np.isfinite(int8).all():
        raise RuntimeError("candidate did not produce a finite waveform matching the FP32 shape")
    relative_rms_error = float(np.sqrt(np.mean((fp32 - int8) ** 2)) / np.sqrt(np.mean(fp32**2)))
    correlation = float(np.corrcoef(fp32, int8)[0, 1])
    return relative_rms_error, correlation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="file name inside --input-dir (default: all 5 models)")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--input-dir", type=Path, default=FP32_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_DIR / "int8")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.model or MODEL_NAMES:
        source = args.input_dir / name
        target = args.output_dir / name.replace(".onnx", ".int8.onnx")
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")

        vocab = vocab_size(source)
        started = time.monotonic()
        quantize_static(
            str(source),
            str(target),
            SyntheticTokenReader(vocab, args.samples),
            quant_format=QuantFormat.QOperator,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv"],
            nodes_to_quantize=decoder_conv_nodes(source),
            per_channel=True,
        )
        relative_rms_error, correlation = compare_audio(source, target, vocab)
        reduction = 100 * (1 - target.stat().st_size / source.stat().st_size)
        print(
            f"{name}: {source.stat().st_size:,} -> {target.stat().st_size:,} bytes "
            f"({reduction:.1f}% smaller), relative RMS error {relative_rms_error:.3f}, "
            f"correlation {correlation:.3f}, {time.monotonic() - started:.1f}s"
        )


if __name__ == "__main__":
    main()
