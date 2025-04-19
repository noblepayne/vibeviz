{
  description = "Audio visualizer that converts audio streams to video with real-time visualization";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    systems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    pkgsForSystem = system: nixpkgs.legacyPackages.${system};
    forAllSystems = fn: (nixpkgs.lib.genAttrs systems (system: (fn (pkgsForSystem system))));
  in {
    formatter = forAllSystems (pkgs: pkgs.alejandra);
    packages = forAllSystems (pkgs: {
      default = pkgs.stdenv.mkDerivation {
        pname = "vibeviz";
        version = "0.1";
        src = ./.;
        nativeBuildInputs = [pkgs.pkg-config];
        buildInputs = [
          pkgs.ffmpeg
          pkgs.fftwFloat
          pkgs.libjpeg
          pkgs.libpng
        ];
        buildPhase = ''
          runHook preBuild
          g++ vibeviz.cpp -o vibeviz \
            $(pkg-config --cflags --libs libavformat libavcodec libavutil libswscale libswresample libjpeg libpng) \
            -lfftw3f \
            -O2 -lpthread -ldl
          runHook postBuild
        '';
        installPhase = ''
          runHook preInstall
          install -Dm755 vibeviz $out/bin/vibeviz
          runHook postInstall
        '';
        meta = {
          description = "Audio visualizer that converts audio streams to video.";
          license = pkgs.lib.licenses.mit;
          platforms = systems;
        };
      };
    });

    # TODO: untested
    devShells = forAllSystems (pkgs: {
      default = pkgs.mkShell {
        buildInputs = [
          pkgs.pkg-config
          pkgs.ffmpeg
          pkgs.fftwFloat
          pkgs.libjpeg
          pkgs.libpng
        ];
      };
    });
  };
}
