defmodule AxeraNPU.Model do
  @moduledoc """
  A model loaded onto the NPU, from `AxeraNPU.load/1`, with its inputs and
  outputs in order. It holds NPU memory until it is garbage collected.
  """

  @enforce_keys [:ref, :inputs, :outputs]
  defstruct [:ref, :inputs, :outputs]

  @type t :: %__MODULE__{
          ref: reference(),
          inputs: [AxeraNPU.Tensor.t()],
          outputs: [AxeraNPU.Tensor.t()]
        }
end
