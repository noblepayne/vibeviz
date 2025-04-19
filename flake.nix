{
  description = "Audio visualizer that converts audio streams to video with real-time visualization";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: {
    formatter.x86_64-linux = nixpkgs.legacyPackages.x86_64-linux.alejandra;

    packages.x86_64-linux.default = let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in
      pkgs.stdenv.mkDerivation {
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
      };
  };
}
