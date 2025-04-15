{
  description = "PipeWire/OpenGL FFT Audio Visualizer";

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
        pipewire = pkgs.pipewire;
        glfw = pkgs.glfw;
        fftw = pkgs.fftwFloat;
        libGL = pkgs.libGL;
        stdenv = pkgs.stdenv;
        src = ./src.cpp;
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "pipewire-fft-vis";
          src = src;
          dontUnpack = true;
          buildInputs = [
            pipewire
            fftw
            glfw
            libGL
          ];
          NIX_CFLAGS_COMPILE = ''
            -I${pipewire.dev}/include/pipewire-0.3
            -I${pipewire.dev}/include/spa-0.2
            -I${fftw.dev}/include
            -I${glfw}/include
          '';
          buildPhase = ''
            g++ -o pipewire-fft-vis $src \
              -Wall -O2 \
              -I${pipewire.dev}/include/pipewire-0.3 \
              -I${pipewire.dev}/include/spa-0.2 \
              -I${fftw.dev}/include \
              -I${glfw}/include \
              -L${pipewire.out}/lib -lpipewire-0.3 \
              -L${fftw.out}/lib -lfftw3f \
              -L${glfw.out}/lib -lglfw \
              -L${libGL.out}/lib -lGL \
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
