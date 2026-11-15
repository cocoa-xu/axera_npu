# :models tests need .axmodel files in the directory AXERA_NPU_MODELS names
# (mobilenetv2, yolov5s, ch_PP_OCRv3_det_npu, ch_PP_OCRv4_rec_npu); :npu
# tests need the NPU (/dev/npu).
models = System.get_env("AXERA_NPU_MODELS")

exclude =
  if(models && File.dir?(models), do: [], else: [:models]) ++
    if File.exists?("/dev/npu"), do: [:no_npu], else: [:npu]

ExUnit.start(exclude: exclude)
