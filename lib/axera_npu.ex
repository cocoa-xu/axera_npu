defmodule AxeraNPU do
  @moduledoc """
  Runs Pulsar2 models (`.axmodel`) on the NPU of Axera AX620E chips, through
  the open `ax620e-npu` driver and an open runtime: no vendor libraries.

      {:ok, model} = AxeraNPU.load("/data/mobilenetv2.axmodel")

      model.inputs
      #=> [%AxeraNPU.Tensor{name: "input", type: {:u, 8}, shape: {1, 224, 224, 3}, size: 150528}]

      {:ok, [scores]} = AxeraNPU.run(model, [rgb])

  Inputs are binaries of the inputs' sizes, or Nx tensors of their types and
  shapes; the outputs come back in the same form. A model runs one call at a
  time; separate models run concurrently, the NPU taking their work in turn.
  """

  alias AxeraNPU.{Model, NIF, Tensor}

  @compile {:no_warn_undefined, Nx}

  @typedoc "An input or output: its bytes, or an Nx tensor."
  @type data :: binary() | struct()

  @doc """
  Loads a model file onto the NPU.
  """
  @spec load(Path.t()) :: {:ok, Model.t()} | {:error, File.posix() | String.t()}
  def load(path) do
    with {:ok, axmodel} <- File.read(path), do: load_binary(axmodel)
  end

  @doc """
  Loads a model from its bytes onto the NPU.
  """
  @spec load_binary(binary()) :: {:ok, Model.t()} | {:error, String.t()}
  def load_binary(axmodel) when is_binary(axmodel) do
    case NIF.model_load(axmodel) do
      {:ok, ref, {inputs, outputs}} ->
        {:ok, %Model{ref: ref, inputs: tensors(inputs), outputs: tensors(outputs)}}

      {:error, _message} = error ->
        error
    end
  end

  @doc """
  A model's inputs and outputs, read from its bytes without the NPU (so on
  any computer).
  """
  @spec describe(binary()) ::
          {:ok, %{inputs: [Tensor.t()], outputs: [Tensor.t()]}} | {:error, String.t()}
  def describe(axmodel) when is_binary(axmodel) do
    case NIF.model_describe(axmodel) do
      {:ok, {inputs, outputs}} -> {:ok, %{inputs: tensors(inputs), outputs: tensors(outputs)}}
      {:error, _message} = error -> error
    end
  end

  @doc """
  Runs a model on its inputs, in order.

  The outputs are Nx tensors if any input is one, binaries otherwise. Raises
  `ArgumentError` for inputs that do not fit the model; returns an error when
  the NPU fails (`:etimedout`, `:eio`, ...).
  """
  @spec run(Model.t(), [data()]) :: {:ok, [data()]} | {:error, atom()}
  def run(%Model{} = model, inputs) when is_list(inputs) do
    if length(inputs) != length(model.inputs) do
      raise ArgumentError,
            "the model takes #{length(model.inputs)} inputs, #{length(inputs)} given"
    end

    binaries = Enum.zip_with(model.inputs, inputs, &to_binary/2)

    case NIF.model_run(model.ref, binaries) do
      {:ok, outputs} ->
        if Enum.any?(inputs, &nx_tensor?/1),
          do: {:ok, Enum.zip_with(model.outputs, outputs, &to_nx/2)},
          else: {:ok, outputs}

      {:error, _reason} = error ->
        error
    end
  end

  defp tensors(list), do: Enum.map(list, &Tensor.from_nif/1)

  defp nx_tensor?(%{__struct__: Nx.Tensor}), do: true
  defp nx_tensor?(_data), do: false

  defp to_binary(%Tensor{size: size}, data) when is_binary(data) and byte_size(data) == size,
    do: data

  defp to_binary(%Tensor{} = tensor, data) when is_binary(data) do
    raise ArgumentError,
          "input #{tensor.name} takes #{tensor.size} bytes, #{byte_size(data)} given"
  end

  defp to_binary(%Tensor{} = tensor, %{__struct__: Nx.Tensor} = data) do
    if Nx.type(data) == tensor.type and Nx.shape(data) == tensor.shape do
      Nx.to_binary(data)
    else
      raise ArgumentError,
            "input #{tensor.name} takes a #{inspect(tensor.type)} tensor of shape " <>
              "#{inspect(tensor.shape)}, #{inspect(Nx.type(data))} of shape " <>
              "#{inspect(Nx.shape(data))} given"
    end
  end

  defp to_binary(%Tensor{} = tensor, data) do
    raise ArgumentError,
          "input #{tensor.name} takes a binary or an Nx tensor, got #{inspect(data)}"
  end

  defp to_nx(%Tensor{type: type, shape: shape}, binary) do
    binary |> Nx.from_binary(type) |> Nx.reshape(shape)
  end
end
