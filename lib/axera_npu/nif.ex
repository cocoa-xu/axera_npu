defmodule AxeraNPU.NIF do
  @moduledoc false

  @on_load :load_nif

  def load_nif do
    :code.priv_dir(:axera_npu)
    |> :filename.join(~c"axera_npu_nif")
    |> :erlang.load_nif(0)
  end

  def model_load(_axmodel), do: :erlang.nif_error(:not_loaded)
  def model_describe(_axmodel), do: :erlang.nif_error(:not_loaded)
  def model_run(_model, _inputs), do: :erlang.nif_error(:not_loaded)
  def ocr_load(_det, _rec, _keys), do: :erlang.nif_error(:not_loaded)
  def ocr_read(_ocr, _rgb, _width, _height), do: :erlang.nif_error(:not_loaded)
end
