defmodule AxeraNPU.OCR.Line do
  @moduledoc """
  A line of text found in an image: its text, its box `{x0, y0, x1, y1}`
  (inclusive pixel coordinates), the detector's score for the box and the
  recognizer's confidence in the text (both from 0 to 1).
  """

  @enforce_keys [:text, :box, :score, :confidence]
  defstruct [:text, :box, :score, :confidence]

  @type t :: %__MODULE__{
          text: String.t(),
          box: {non_neg_integer(), non_neg_integer(), non_neg_integer(), non_neg_integer()},
          score: float(),
          confidence: float()
        }

  @doc false
  @spec from_nif({String.t(), tuple(), float(), float()}) :: t()
  def from_nif({text, box, score, confidence}) do
    %__MODULE__{text: text, box: box, score: score, confidence: confidence}
  end
end
