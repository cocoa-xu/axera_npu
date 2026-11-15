defmodule AxeraNPU.OCRTest do
  use ExUnit.Case, async: true

  alias AxeraNPU.OCR

  test "images must be RGB of their size" do
    ocr = %OCR{ref: make_ref()}

    assert_raise ArgumentError, ~r/a 4 × 2 RGB image is 24 bytes, 3 given/, fn ->
      OCR.read(ocr, "abc", 4, 2)
    end

    assert_raise ArgumentError, ~r/shape \{height, width, 3\}, \{:u, 8\} of shape \{2, 4\}/, fn ->
      OCR.read(ocr, Nx.broadcast(Nx.tensor(0, type: :u8), {2, 4}))
    end
  end

  test "missing files are file errors" do
    assert {:error, :enoent} =
             OCR.load("/nonexistent/det", "/nonexistent/rec", "/nonexistent/keys")
  end
end
