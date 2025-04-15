{
  description = "A simple audio decoder flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    formatter.x86_64-linux = nixpkgs.legacyPackages.x86_64-linux.alejandra;
    
    packages.x86_64-linux.default = let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in pkgs.stdenv.mkDerivation {
      pname = "audio-decoder";
      version = "0.1";
      src = ./.;

      nativeBuildInputs = [ pkgs.pkg-config ];

      buildInputs = [
        pkgs.ffmpeg
        pkgs.fftwFloat
      ];

      buildPhase = ''
        runHook preBuild
        g++ viz.cpp -o audio-decoder \
          $(pkg-config --cflags --libs libavformat libavcodec libavutil libswscale libswresample) \
          -lfftw3f \
          -O2 -lpthread -ldl
        runHook postBuild
      '';

      installPhase = ''
        runHook preInstall
        install -Dm755 audio-decoder $out/bin/audio-decoder
        runHook postInstall
      '';
    };
  };
}

