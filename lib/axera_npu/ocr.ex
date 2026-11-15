defmodule AxeraNPU.OCR do
  @moduledoc """
  Finds and reads the lines of text in an image with PP-OCR models on the
  NPU: a DB detector (`ch_PP_OCRv3_det` at 640 × 480) and a CTC recognizer
  (`ch_PP_OCRv4_rec` at 320 × 48), with the recognizer's dictionary.

      {:ok, ocr} =
        AxeraNPU.OCR.load("ch_PP_OCRv3_det_npu.axmodel", "ch_PP_OCRv4_rec_npu.axmodel",
          "ppocr_keys_v1.txt")

      {:ok, lines} = AxeraNPU.OCR.read(ocr, rgb, 960, 540)
      Enum.map(lines, & &1.text)
      #=> ["NanoKVM-Go offline OCR on the NPU", "离线文字识别：神经网络处理器", ...]

  Lines are axis-aligned boxes, which suits screens. A line wider than the
  recognizer's input is read in pieces, cut between letters.
  """

  alias AxeraNPU.NIF
  alias AxeraNPU.OCR.Line

  @compile {:no_warn_undefined, Nx}

  @enforce_keys [:ref]
  defstruct [:ref]

  @type t :: %__MODULE__{ref: reference()}

  @doc """
  Loads the detector and the recognizer onto the NPU, with the recognizer's
  dictionary (one character per line).
  """
  @spec load(Path.t(), Path.t(), Path.t()) :: {:ok, t()} | {:error, File.posix() | String.t()}
  def load(detector, recognizer, dictionary) do
    with {:ok, det} <- File.read(detector),
         {:ok, rec} <- File.read(recognizer),
         {:ok, keys} <- File.read(dictionary) do
      load_binary(det, rec, keys)
    end
  end

  @doc """
  Like `load/3`, from the files' contents.
  """
  @spec load_binary(binary(), binary(), binary()) :: {:ok, t()} | {:error, String.t()}
  def load_binary(detector, recognizer, dictionary)
      when is_binary(detector) and is_binary(recognizer) and is_binary(dictionary) do
    case NIF.ocr_load(detector, recognizer, dictionary) do
      {:ok, ref} -> {:ok, %__MODULE__{ref: ref}}
      {:error, _message} = error -> error
    end
  end

  @doc """
  The lines of text in an RGB image (`width × height × 3` bytes), top to
  bottom and left to right.
  """
  @spec read(t(), binary(), pos_integer(), pos_integer()) :: {:ok, [Line.t()]} | {:error, atom()}
  def read(%__MODULE__{ref: ref}, rgb, width, height)
      when is_binary(rgb) and is_integer(width) and width > 0 and is_integer(height) and
             height > 0 do
    if byte_size(rgb) != width * height * 3 do
      raise ArgumentError,
            "a #{width} × #{height} RGB image is #{width * height * 3} bytes, " <>
              "#{byte_size(rgb)} given"
    end

    case NIF.ocr_read(ref, rgb, width, height) do
      {:ok, lines} -> {:ok, Enum.map(lines, &Line.from_nif/1)}
      {:error, _reason} = error -> error
    end
  end

  @doc """
  The lines of text in an RGB image as an Nx tensor of type `{:u, 8}` and
  shape `{height, width, 3}`.
  """
  @spec read(t(), struct()) :: {:ok, [Line.t()]} | {:error, atom()}
  def read(%__MODULE__{} = ocr, %{__struct__: Nx.Tensor} = image) do
    case {Nx.type(image), Nx.shape(image)} do
      {{:u, 8}, {height, width, 3}} ->
        read(ocr, Nx.to_binary(image), width, height)

      {type, shape} ->
        raise ArgumentError,
              "an image is a {:u, 8} tensor of shape {height, width, 3}, " <>
                "#{inspect(type)} of shape #{inspect(shape)} given"
    end
  end
end
