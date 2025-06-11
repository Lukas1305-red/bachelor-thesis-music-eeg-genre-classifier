{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  # Selects the CUDA toolkit version 12.3 from the available CUDA packages in Nixpkgs.
  # The `cudaPackages` attribute provides access to the full suite of CUDA tools and libraries for this version.
  # To use a different CUDA version, replace `cudaPackages_12_3` with another version (e.g., `cudaPackages_11_8`).
  # Individual components (such as `cuda`, `cudatoolkit`, `cudnn`, etc.) can be accessed as attributes of `cudaPackages`
  # and added to the `packages` list in your devenv shell as needed.
  # Example: `cudaPackages.cudatoolkit`, `cudaPackages.cudnn`, etc.
  cudaPackages = pkgs.cudaPackages_12_4;
  # cudaPackages = pkgs.cudaPackages_10_0
  # cudaPackages = pkgs.cudaPackages_10_1
  # cudaPackages = pkgs.cudaPackages_10_2
  # cudaPackages = pkgs.cudaPackages_10
  # cudaPackages = pkgs.cudaPackages_11_0
  # cudaPackages = pkgs.cudaPackages_11_1
  # cudaPackages = pkgs.cudaPackages_11_2
  # cudaPackages = pkgs.cudaPackages_11_3
  # cudaPackages = pkgs.cudaPackages_11_4
  # cudaPackages = pkgs.cudaPackages_11_5
  # cudaPackages = pkgs.cudaPackages_11_6
  # cudaPackages = pkgs.cudaPackages_11
in {
  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages = with pkgs; [
    cudaPackages.cuda_nvcc
    git
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.11";
    venv.enable = true;
    uv = {
      enable = true;
      sync = {
        enable = true;
        extras = ["build"];
      };
    };
    manylinux.enable = false;
    libraries = with pkgs; [
      zlib
    ];
  };

  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts = {
    reinstall-venv.exec = "rm -rf $DEVENV_ROOT/.venv $DEVENV_ROOT/.direnv $DEVENV_ROOT/.devenv && direnv reload";
  };

  enterShell = ''
    uv sync --extra build --extra compile
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  # enterTest = ''
  #   echo "Running tests"
  #   git --version | grep --color=auto "${pkgs.git.version}"
  # '';

  # https://devenv.sh/git-hooks/
  git-hooks.hooks = {
    shellcheck.enable = true;
    alejandra.enable = true;
    ruff.enable = true;
    ruff-format.enable = true;
  };
  # See full reference at https://devenv.sh/reference/options/
}
