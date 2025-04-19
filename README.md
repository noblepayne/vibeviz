# vibeviz

Audio visualizer that converts audio streams to video with real-time visualization. Supports Icecast, MP3, M4A inputs and outputs to MP4/MKV files or RTMP streams.

## Features

- Real-time audio spectrum visualization
- Synthwave-style color palette
- Background image support (JPEG/PNG)
- Output to file (MP4/MKV) or RTMP stream
- Smooth animations and peak detection
- Audio passthrough with AAC encoding

## Requirements

- FFmpeg libraries
- FFTW3
- libjpeg/libpng (for background images)
- C++17 compiler

## Build

```sh
nix build
```

## Usage

```sh
vibeviz <input_audio> <output_path> [background_image.jpg|.png]
```

Examples:

```sh
# Local file output
vibeviz input.mp3 output.mp4 background.jpg

# RTMP stream
vibeviz http://icecast.example.com/stream.mp3 rtmp://live.twitch.tv/app/streamkey
```

## License

MIT
