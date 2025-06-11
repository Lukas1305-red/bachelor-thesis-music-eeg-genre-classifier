# devenv-cli

This folder contains a reproducible development shell powered by [devenv](https://devenv.sh/). It provides a consistent environment for CLI development, including system packages, language runtimes, and Python dependency management with [uv](https://github.com/astral-sh/uv).

## Features

- Isolated shell environment
- Easily extendable with system packages and languages
- Fast Python dependency management via `uv` and `pyproject.toml`/`uv.lock`
- Automatic environment activation with [direnv](https://direnv.net/)

## Getting Started

1. **Install [Nix](https://nixos.org/download.html), [devenv](https://devenv.sh/getting-started/), and [direnv](https://direnv.net/docs/installation.html)**
2. Allow direnv in this directory (run once):

    ```sh
    direnv allow
    ```

## Manage Python Dependencies with uv (pyproject.toml + uv.lock)

1. In your `devenv.nix`, enable Python, virtual environments, and `uv`:

    ```nix
    languages.python = {
      enable = true;
      venv.enable = true;
      uv = {
        enable = true;
        sync.enable = true;
      };
    };
    ```

2. Add your dependencies to `pyproject.toml` under the `[project]` section, for example:

    ```toml
    [project]
    dependencies = [
      "requests",
      "numpy"
    ]
    ```

3. Use `uv` to resolve and lock dependencies:

    ```sh
    uv lock
    ```

    This will generate or update `uv.lock`.

4. To install dependencies from the lock file and sync them with the virtual environment:

    ```sh
    uv sync
    ```

5. To add a new dependency:

    ```sh
    uv add <package>
    ```

    This updates both `pyproject.toml` and `uv.lock`.

## Extending the Environment

### Add System Packages

Edit `devenv.nix` and add packages to the `packages` list:

```nix
packages = with pkgs; [
  git
  curl
  # Add more packages here
];
```

Find packages provided by your inputs (`devenv.yaml`) with `devenv search`.

### Enable Languages

Enable language support by adding to the `languages` section:

```nix
languages.python.enable = true;
languages.nodejs.enable = true;
# Add more languages as needed
```

For a comprehensive list of available language options and their configuration details,
please refer to the official devenv documentation at: <https://devenv.sh/languages/>

## Installing flash-attention

**THIS IS ALREADY IMPLEMENTED IN THE TEMPLATE**. However, if you want to apply it elsewhere, here is the rundown:

To use flash-attention, both `torch` and the CUDA compiler (`nvcc`) must be available during build time.

1. **Ensure `nvcc` is available** by adding it to your `devenv.nix`:

    Edit your `devenv.nix` to include the correct CUDA version. For example, to use CUDA 12.6:

    ```nix
    let
    # ...
    cudaPackages = pkgs.cudaPackages_12_4;
    # ...
    in
    # ...
    packages = with pkgs; [
      cudaPackages.cuda_nvcc
      git
    ];
    ```

    If you need a different CUDA version (e.g., 11.8), change the line to:

    ```nix
    cudaPackages = pkgs.cudaPackages_11_8;
    ```

    **Make sure the CUDA version here matches the CUDA version of your `torch` build** (e.g., `torch==2.6.0+cu118` for CUDA 11.8).

2. **Configure Python and uv in `devenv.nix`**  
    Make sure your `devenv.nix` includes the following to enable Python, uv, and syncing of the `build` and `compile` extras:

    ```nix
    languages.python = {
      enable = true;
      version = "3.12";
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

    enterShell = ''
      uv sync --extra build --extra compile
    '';
    ```

    - The `uv.sync.extras = ["build"]` ensures the `build` group is always synced.
    - The `enterShell` hook ensures both `build` and `compile` extras are installed when you enter the shell.
    - By enabling the `build` extra via devenv. `uv sync --extra build` will be run before the shell hook, installing flash-attn's build dependencies.

3. **Add dependencies to `pyproject.toml`**  
    Your `pyproject.toml` should include:

    ```toml
    [project]
    dependencies = []

    [project.optional-dependencies]
    build = ["torch~=2.6.0", "setuptools", "packaging", "psutil", "numpy"]
    compile = ["flash-attn~=2.7.4"]

    [tool.uv]
    no-build-isolation-package = ["flash-attn"]

    [[tool.uv.dependency-metadata]]
    name = "flash-attn"
    requires-dist = ["torch", "einops", "psutil", "numpy"]
    ```

    - The `build` group includes `torch` and related build tools.
    - The `compile` group includes `flash-attn`.

Now, both `torch` and `flash-attn` will be reproducibly managed by `uv` and installed in your environment whenever you enter the shell.

## Resources

- [devenv documentation](https://devenv.sh/)
- [uv documentation](https://github.com/astral-sh/uv)
- [direnv documentation](https://direnv.net/)
- [flash-attn via uv](https://docs.astral.sh/uv/concepts/projects/config/#build-isolation)
