# PP-OCRv5 for `AxeraNPU.OCR`

How the PP-OCRv5 mobile models that `AxeraNPU.OCR` reads with are compiled
for the AX620E: text detection at 640 × 640 and recognition at 48 × 320, both
taking u8 RGB in NHWC order, with the normalization and the RGB-to-BGR
swap inside the model.

| Model | File | Size | On the NanoKVM-Go+ |
| --- | --- | --- | --- |
| Detection, U8 activations | `det.axmodel` | 1.6 MB | 32 ms |
| Recognition, U16 activations | `rec.axmodel` | 9.9 MB | 23 ms a piece |

A screen page of six lines reads in about half a second. AXERA's own
AX620E detection model works at 960 × 960 with U16 activations and needs
51 MiB of NPU memory for them, more than the Go+'s 64 MiB pool leaves.

## Sources

- The ONNX models, simplified to static shapes, from
  [AXERA-TECH/PPOCR_v5](https://huggingface.co/AXERA-TECH/PPOCR_v5) (MIT):
  `onnx/det_mobile_sim_static.onnx` and `onnx/rec_mobile_sim_static.onnx`,
  and the recognizer's dictionary `ppocrv5_dict.txt`. PP-OCRv5 itself is
  Apache-2.0, from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
- The calibration images of
  [AXERA-TECH/PPOCR_v5.axera](https://github.com/AXERA-TECH/PPOCR_v5.axera)
  release v1.0.1: `ocr_pic.tar.gz` and `rec_pic.tar.gz`.
- Pulsar2 4.2, the container image `ax_pulsar2_4.2.tar.gz` of
  [AXERA-TECH/Pulsar2](https://huggingface.co/AXERA-TECH/Pulsar2).

## Building

With the files above in one directory, next to `det640_u8.json`,
`rec_u16.json` and `build.sh` from here, on x86-64 Linux with Podman (or
Docker):

```sh
podman load -i ax_pulsar2_4.2.tar.gz
mkdir -p out
podman run --rm -v "$PWD":/data --entrypoint /bin/bash pulsar2:4.2 /data/build.sh
```

`build.sh` sets the detection model's input to 640 × 640 (its `Resize`
nodes scale by factors, so nothing else depends on the size), then builds
`out/det640_u8/det.axmodel` and `out/rec_u16/rec.axmodel`.
