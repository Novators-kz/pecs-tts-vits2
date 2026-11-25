# Model files

The `.onnx` weights are not in git (too large). They live in a private
Cloudflare R2 bucket named `pecs-tts-models`. The configs (`config_*.json`) are in this repo.

The credentials are not in this repo. They are the R2 access key pair for the
company Cloudflare account (see the backend `.env`, `R2_*` variables, and the
company password manager).

## Bucket contents

| Key | Size (bytes) | Use |
| --- | ---: | --- |
| `fp32/girl/vits2_eng_girl.onnx` | 117,334,902 | Web, deployed |
| `fp32/girl/vits2_kaz_girl.onnx` | 156,630,415 | Web, deployed |
| `fp32/girl/vits2_rus_girl.onnx` | 156,623,502 | Web, deployed |
| `fp32/nao/vits2_kaz_nao_4000.onnx` | 156,651,502 | Web, deployed as `vits2_kaz_nao.onnx` |
| `fp32/nao/vits2_rus_nao_4000.onnx` | 156,644,590 | Web, deployed as `vits2_rus_nao.onnx` |
| `fp32/nao/vits2_{kaz,rus}_nao_{3000,5000}.onnx` | 156 MB each | Other training steps, not deployed |
| `int8/girl/vits2_{eng,kaz,rus}_girl.int8.onnx` | 82 / 122 / 122 MB | Mobile source files |
| `int8/nao/vits2_{kaz,rus}_nao_4000.int8.onnx` | 122 MB each | Mobile source files |
| `manifest.json` | small | Every key with its size and SHA-256 |

The mobile app does not read this bucket. It downloads INT8 packs through the
backend from the separate bucket `pecs-tts-voice-packs` (`tts/v1/...`). Do not
change that bucket.

## Restore the web models on a new VPS

The service reads `vits2_<name>.onnx` from `TTS_MODEL_DIR`. The two NAO files
must be renamed (step 4000 is the one in use).

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=auto
EP=https://<account-id>.r2.cloudflarestorage.com   # R2_ENDPOINT_URL
B=s3://pecs-tts-models
mkdir -p models && cd models
for n in eng kaz rus; do
  aws s3 cp $B/fp32/girl/vits2_${n}_girl.onnx . --endpoint-url $EP
done
aws s3 cp $B/fp32/nao/vits2_kaz_nao_4000.onnx vits2_kaz_nao.onnx --endpoint-url $EP
aws s3 cp $B/fp32/nao/vits2_rus_nao_4000.onnx vits2_rus_nao.onnx --endpoint-url $EP
```

Check the files against `manifest.json` (`aws s3 cp $B/manifest.json .`) with
`sha256sum`. Then mount the folder into the container and set `TTS_MODEL_DIR`.

## Quantization

`tools/quantize_static.py` makes INT8 files from the FP32 files. See its
`--help`. Install `tools/requirements.txt` first (Python 3.11).
