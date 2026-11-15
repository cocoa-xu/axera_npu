defmodule AxeraNPU.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/cocoa-xu/axera_npu"

  def project do
    [
      app: :axera_npu,
      version: @version,
      elixir: "~> 1.15",
      description: description(),
      package: package(),
      compilers: [:elixir_make | Mix.compilers()],
      make_targets: ["all"],
      make_clean: ["clean"],
      make_error_message: make_error_message(),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      source_url: @source_url,
      dialyzer: [
        flags: [:missing_return, :extra_return, :unmatched_returns, :error_handling],
        plt_add_apps: [:nx]
      ]
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  def cli do
    [preferred_envs: [docs: :docs, "hex.build": :docs, dialyzer: :test, credo: :test]]
  end

  defp description do
    "Runs Pulsar2 models (.axmodel) on the NPU of Axera AX620E chips through an open " <>
      "driver and runtime, with binaries or Nx tensors, and reads text with PP-OCR models."
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url},
      files: [
        "lib",
        "c_src/*.[ch]",
        "Makefile",
        "mix.exs",
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        ".formatter.exs"
      ]
    ]
  end

  defp deps do
    [
      {:elixir_make, "~> 0.9", runtime: false},
      {:nx, "~> 0.9 or ~> 1.0", optional: true},
      {:ex_doc, "~> 0.34", only: :docs, runtime: false},
      {:dialyxir, "~> 1.4", only: :test, runtime: false},
      {:credo, "~> 1.7", only: :test, runtime: false}
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "CHANGELOG.md", "LICENSE"],
      source_ref: "v#{@version}",
      groups_for_modules: [
        Models: [AxeraNPU, AxeraNPU.Model, AxeraNPU.Tensor],
        OCR: [AxeraNPU.OCR, AxeraNPU.OCR.Line]
      ]
    ]
  end

  defp make_error_message do
    """
    Could not build the axera_npu NIF. It needs a C toolchain and the ERTS
    headers, which elixir_make (and Nerves, for the target) provide.
    """
  end
end
