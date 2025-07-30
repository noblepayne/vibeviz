# vibeviz Testing Setup

Manual integration testing setup for the vibeviz tool using local streaming infrastructure.

## Overview

This setup creates a complete audio streaming pipeline:
`MP3 file → HTTP stream → vibeviz → RTMP server → mpv viewer`

## Prerequisites

- Nix with flakes enabled
- An MP3 file for testing (referenced as `sample.mp3` below)

## Setup Steps

### 1. Start the MediaMTX RTMP server

Create a config file `mediamtx.yml`:
```yaml
paths:
  test1:
```

Run MediaMTX:
```bash
nix run nixpkgs#mediamtx
```

*Note: MediaMTX will auto-discover `mediamtx.yml` in the current directory, or you can specify the config path as the first argument.*

### 2. Start vibeviz

```bash
nix run . -- http://localhost:7777 rtmp://localhost:1935/test1
```

*Alternative: Use `vibeviz` instead of `nix run . --` if you have it in your PATH.*

### 3. Stream test audio over HTTP

```bash
while true; do
  nix run nixpkgs#ffmpeg -- -re -i sample.mp3 -c copy -f mp3 -listen 1 http://0.0.0.0:7777
  sleep 1
done
```

### 4. View the output stream

```bash
while true; do
  nix run nixpkgs#mpv rtmp://localhost/test1
  sleep 1
done
```

## Testing Flow

1. MediaMTX provides RTMP server on port 1935
2. ffmpeg serves MP3 file as HTTP stream on port 7777
3. vibeviz reads HTTP audio stream from port 7777 and outputs to RTMP
4. mpv connects to RTMP server to view the processed stream

## Notes

- This is a manual integration testing setup
- The `while true` loops handle connection drops and restarts
- Future improvements could include proper nix devshell integration
