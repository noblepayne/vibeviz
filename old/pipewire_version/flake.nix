{
  description = "PipeWire/OpenGL FFT Audio Visualizer (Refactored)";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
      in {
        formatter = pkgs.alejandra;
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "pipewire-fft-vis";
          version = "1.0.1";
          src = ./src.cpp;
          dontUnpack = true;
          buildInputs = [
            pkgs.pipewire
            pkgs.fftwFloat
            pkgs.glfw
            pkgs.libGL
          ];
          NIX_CFLAGS_COMPILE = ''
            -I${pkgs.pipewire.dev}/include/pipewire-0.3
            -I${pkgs.pipewire.dev}/include/spa-0.2
            -I${pkgs.fftwFloat.dev}/include
            -I${pkgs.glfw}/include
          '';
          buildPhase = ''
            g++ -o pipewire-fft-vis $src \
              -Wall -O2 \
              -I${pkgs.pipewire.dev}/include/pipewire-0.3 \
              -I${pkgs.pipewire.dev}/include/spa-0.2 \
              -I${pkgs.fftwFloat.dev}/include \
              -I${pkgs.glfw}/include \
              -L${pkgs.pipewire.out}/lib -lpipewire-0.3 \
              -L${pkgs.fftwFloat.out}/lib -lfftw3f \
              -L${pkgs.glfw.out}/lib -lglfw \
              -L${pkgs.libGL.out}/lib -lGL \
              -lpthread \
              -ldl
          '';
          installPhase = ''
            mkdir -p $out/bin
            cp pipewire-fft-vis $out/bin/
          '';
        };
        apps.default = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
        };
      }
    );
}
