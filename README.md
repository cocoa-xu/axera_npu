# AxeraNPU

Runs Pulsar2 models (`.axmodel`) on the NPU of Axera AX620E chips from
Elixir, through an open driver and an open runtime: no vendor libraries on
the device. Inputs and outputs are binaries or Nx tensors, and
`AxeraNPU.OCR` reads text with PP-OCR models.

| Part | Where | License |
| --- | --- | --- |
| The Elixir API and its NIF | `lib/`, `c_src/axera_npu_nif.c` | Apache-2.0 |
| The runtime: reads `.axmodel` (ONNX) files and runs them through `/dev/npu` | `c_src/axnpu.[ch]` | Apache-2.0 |
| PP-OCR text detection and recognition | `c_src/ocr.[ch]` | Apache-2.0 |
| The `ax620e-npu` driver, with the vendor's `/dev/npu` interface | `kernel/` | GPL-2.0-only |
| A Buildroot package for the driver | `buildroot/` | Apache-2.0 |
| `axrun`, `ocr` and `neu.py`, to try things outside the BEAM | `tools/` | Apache-2.0 |

## Hardware

It runs on the Sipeed NanoKVM-Go+ (AX620QE, 32-bit mainline Linux 7.3).
The other AX620E chips (AX620Q, AX630C) have the same NPU; the AX630C boards
(NanoKVM Pro, MaixCam2, M5Stack LLM630) run 64-bit kernels, and the driver
speaks only the 32-bit form of the interface so far.

The NPU has no IOMMU, so its buffers come from a memory pool of their own.
The driver needs the NPU's device tree node and that pool:

```dts
reserved-memory {
	npu_memory: npu-memory {
		compatible = "shared-dma-pool";
		reusable;
		size = <0x04000000>;
	};
};

npu@3800000 {
	compatible = "axera,ax620e-npu";
	reg = <0x03800000 0xb0000>, <0x03000000 0x280000>;
	reg-names = "regs", "ocm";
	interrupts = <GIC_SPI 38 IRQ_TYPE_EDGE_RISING>,
		     <GIC_SPI 39 IRQ_TYPE_EDGE_RISING>,
		     <GIC_SPI 42 IRQ_TYPE_EDGE_RISING>,
		     <GIC_SPI 43 IRQ_TYPE_EDGE_RISING>,
		     <GIC_SPI 40 IRQ_TYPE_EDGE_RISING>;
	interrupt-names = "normal0", "normal1", "error0", "error1", "top";
	memory-region = <&npu_memory>;
};
```

## Installation

In a Nerves application:

```elixir
def deps do
  [
    {:axera_npu, github: "cocoa-xu/axera_npu"},
    # optional, for Nx tensors in and out
    {:nx, "~> 0.9"}
  ]
end
```

The Nerves system builds the driver: copy `buildroot/axera-npu` into its
`package` directory, source its `Config.in`, set `BR2_PACKAGE_AXERA_NPU=y`,
and load it with `modprobe ax620e-npu`.

## Usage

```elixir
{:ok, model} = AxeraNPU.load("/data/models/mobilenetv2.axmodel")

model.inputs
#=> [%AxeraNPU.Tensor{name: "input", type: {:u, 8}, shape: {1, 224, 224, 3}, size: 150528}]

{:ok, [scores]} = AxeraNPU.run(model, [rgb])
```

With Nx tensors in, Nx tensors come out:

```elixir
image = rgb |> Nx.from_binary(:u8) |> Nx.reshape({1, 224, 224, 3})
{:ok, [scores]} = AxeraNPU.run(model, [image])
Nx.argmax(scores, axis: 1)
```

`AxeraNPU.describe/1` reads a model's inputs and outputs without the NPU, on
any computer.

### Reading text

```elixir
{:ok, ocr} =
  AxeraNPU.OCR.load("ch_PP_OCRv3_det_npu.axmodel", "ch_PP_OCRv4_rec_npu.axmodel",
    "ppocr_keys_v1.txt")

{:ok, lines} = AxeraNPU.OCR.read(ocr, rgb, 960, 540)
Enum.map(lines, & &1.text)
#=> ["NanoKVM-Go offline OCR on the NPU", "离线文字识别：神经网络处理器", ...]
```

## Performance

On the NanoKVM-Go+, the NPU at 594 MHz:

| Model | Input | On the NPU | Through Elixir |
| --- | --- | --- | --- |
| MobileNetV2 | 224 × 224 | 1.8 ms | 2.1 ms |
| YOLOv5s | 640 × 640 | 26 ms | 47 ms: its 8.5 MB of outputs are copied |
| PP-OCRv3 detection | 640 × 480 | 14 ms | |
| PP-OCRv4 recognition | 320 × 48 | 7.6 ms | |
| OCR of a 960 × 540 page, six lines | | | 260 ms |

## Models

No models come with the library. Compile ONNX models for the AX620E with
AXERA's Pulsar2; MobileNetV2 and YOLOv5s also come with the NPU sample of
[AXERA-TECH/ax620e_bsp_sdk](https://github.com/AXERA-TECH/ax620e_bsp_sdk).
Models keep their own licenses: PP-OCR is Apache-2.0, YOLOv5 is AGPL-3.0.

A model runs when Pulsar2 compiled its whole graph into one NPU program;
graphs that leave operators to the CPU are not supported yet.

## License

Apache-2.0, except the driver in `kernel/`, which is GPL-2.0-only.
