defmodule AxeraNPUTest do
  use ExUnit.Case, async: true

  alias AxeraNPU.{Model, Tensor}

  defp model_path(name), do: Path.join(System.fetch_env!("AXERA_NPU_MODELS"), name <> ".axmodel")

  defp fake_model do
    %Model{
      ref: make_ref(),
      inputs: [%Tensor{name: "input", type: {:u, 8}, shape: {1, 2, 2, 3}, size: 12}],
      outputs: [%Tensor{name: "output", type: {:f, 32}, shape: {1, 4}, size: 16}]
    }
  end

  test "a file that is not a model is refused" do
    assert {:error, "not an ONNX model"} = AxeraNPU.describe("not a model")
    assert {:error, "not an ONNX model"} = AxeraNPU.load_binary("not a model")
  end

  test "a missing file is a file error" do
    assert {:error, :enoent} = AxeraNPU.load("/nonexistent/model.axmodel")
  end

  test "inputs are checked against the model" do
    model = fake_model()

    assert_raise ArgumentError, ~r/takes 1 inputs, 2 given/, fn ->
      AxeraNPU.run(model, [<<0::96>>, <<0::96>>])
    end

    assert_raise ArgumentError, ~r/input input takes 12 bytes, 3 given/, fn ->
      AxeraNPU.run(model, ["abc"])
    end

    assert_raise ArgumentError, ~r/takes a binary or an Nx tensor/, fn ->
      AxeraNPU.run(model, [:image])
    end
  end

  test "Nx inputs must have the input's type and shape" do
    model = fake_model()

    assert_raise ArgumentError, ~r/\{:u, 8\} tensor of shape \{1, 2, 2, 3\}, \{:f, 32\}/, fn ->
      AxeraNPU.run(model, [Nx.broadcast(0.0, {1, 2, 2, 3})])
    end

    assert_raise ArgumentError, ~r/of shape \{2, 2, 3\} given/, fn ->
      AxeraNPU.run(model, [Nx.broadcast(Nx.tensor(0, type: :u8), {2, 2, 3})])
    end
  end

  @tag :models
  test "models are described without the NPU" do
    assert {:ok, %{inputs: [input], outputs: [output]}} =
             AxeraNPU.describe(File.read!(model_path("mobilenetv2")))

    assert input == %Tensor{name: "input", type: {:u, 8}, shape: {1, 224, 224, 3}, size: 150_528}
    assert output == %Tensor{name: "output", type: {:f, 32}, shape: {1, 1000}, size: 4000}

    assert {:ok, %{inputs: [_images], outputs: outputs}} =
             AxeraNPU.describe(File.read!(model_path("yolov5s")))

    assert Enum.map(outputs, & &1.shape) == [{1, 80, 80, 255}, {1, 40, 40, 255}, {1, 20, 20, 255}]
  end

  @tag :models
  @tag :no_npu
  test "without the NPU, loading says so" do
    assert {:error, "/dev/npu: " <> _reason} = AxeraNPU.load(model_path("mobilenetv2"))
  end

  @tag :models
  @tag :npu
  test "a model runs on binaries and on Nx tensors" do
    {:ok, model} = AxeraNPU.load(model_path("mobilenetv2"))
    [input] = model.inputs

    assert {:ok, [scores]} = AxeraNPU.run(model, [:binary.copy(<<128>>, input.size)])
    assert byte_size(scores) == 4000

    image = Nx.broadcast(Nx.tensor(128, type: :u8), input.shape)
    assert {:ok, [tensor]} = AxeraNPU.run(model, [image])
    assert Nx.shape(tensor) == {1, 1000}
    assert Nx.to_binary(tensor) == scores
  end
end
