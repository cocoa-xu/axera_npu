defmodule AxeraNPU.Tensor do
  @moduledoc """
  An input or output of a model: its name, its type and shape as Nx writes
  them (`{:u, 8}`, `{1, 224, 224, 3}`), and its size in bytes.
  """

  @enforce_keys [:name, :type, :shape, :size]
  defstruct [:name, :type, :shape, :size]

  @type t :: %__MODULE__{
          name: String.t(),
          type: {:f | :u | :s, pos_integer()},
          shape: tuple(),
          size: pos_integer()
        }

  @doc false
  @spec from_nif({String.t(), {atom(), pos_integer()}, tuple(), pos_integer()}) :: t()
  def from_nif({name, type, shape, size}) do
    %__MODULE__{name: name, type: type, shape: shape, size: size}
  end
end
