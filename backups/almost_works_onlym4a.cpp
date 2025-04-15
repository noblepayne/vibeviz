PROMPT:
1. Understand the error messages:

   - [aac @ ...] frame_size (1024) was not respected for a non-last frame  
     This means you're not always feeding exactly 1024 samples per audio frame to the AAC encoder (unless it's the very last frame).

   - [aac @ ...] nb_samples (4096) > frame_size (1024)  
     This means you sent more samples than allowed for a single AAC frame (the encoder expects 1024, got 4096).

2. Conclusions/causes:

   - AAC encoding only supports fixed-size input frames (frame_size, typically 1024 for 44.1K) **except possibly for the last frame**.
   - WAV or MP3 decoders can return arbitrary frame sizes (WAV in particular may return multiples of 1024, e.g., 4096).
   - Feeding the decoder frames directly to the encoder (even after resample) means sometimes frames are too large, too small, or not an exact multiple of 1024.

3. Plan to make input agnostic and reliable:

   a. Always buffer decoded audio after resampling/conversion until you have at least 1024 samples (per channel for the encoder's layout).
   b. Feed the encoder only frames with exactly 1024 samples (except possibly the tail).
   c. On end-of-input/flush, send any remaining samples (less than 1024).
   d. For mono/visualization FFT, keep your approach (since you internally manage block sizes there).

4. Implementation steps:

   - Create a circular or append buffer to collect/resample output samples for audio encoding (float or planar float format, according to encoder).
   - When enough samples (>=1024) are in the buffer, feed frames with exactly 1024 to the encoder, advancing the buffer.
   - On flush/end, feed any remaining (tail) samples as a final frame (may be <1024; acceptable as the last frame).
   - Ensure you do not pass more than 1024 samples per frame to AAC encoder.
   - Do resampling (if needed) before buffering for encoder.

5. Result:

   - The program will handle arbitrary input (mp3, wav, any format) and only ever pass legal frame sizes to the encoder.
   - The specific errors regarding frame_size not being respected or too large will be gone.
   - No special-case code is needed for MP3 vs WAV.

6. OUTPUT: output a fixed and complete and total file that is ready to handy mp3 or wav audio cleanly and the rendered video remains tightly synced to the audio.

CODE:

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdarg>
#include <fftw3.h>
#include <vector>
#include <mutex>
#include <algorithm>
#include <atomic>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
}

#define SAMPLE_RATE 44100
#define SAMPLES_PER_FRAME (SAMPLE_RATE / 30) // 1470 samples at 44100Hz/30fps
#define BLOCK_SIZE 1024
#define NUM_BARS 36
#define SMOOTH_FACTOR 0.33f
#define SMOOTH_PEAK 0.13f
#define DB_FLOOR -65.0f
#define DB_CEIL 0.0f

struct VisData {
    float magnitudes[NUM_BARS];
    float target_mags[NUM_BARS];
    float peaks[NUM_BARS];
    float peaks_vel[NUM_BARS];
    float bin_maxes[NUM_BARS];
    float bin_ema[NUM_BARS];
    std::mutex mutex;
};

static float freq_table[NUM_BARS + 1];
static fftwf_plan global_plan = NULL;
static std::mutex fftw_plan_mutex;

static void make_freq_table() {
    const float min_hz = 40.0f, max_hz = 18000.0f;
    for (int i = 0; i <= NUM_BARS; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(NUM_BARS);
        freq_table[i] = min_hz * powf(max_hz / min_hz, t);
    }
}

static int bin_freq_to_fft_bin(float freq) {
    int bin = static_cast<int>(roundf(freq * BLOCK_SIZE / SAMPLE_RATE));
    if (bin < 0) bin = 0;
    if (bin > BLOCK_SIZE / 2) bin = BLOCK_SIZE / 2;
    return bin;
}

static void process_fft(VisData *vis_data, float *samples) {
    static float fft_in[BLOCK_SIZE];
    static fftwf_complex fft_out[BLOCK_SIZE / 2 + 1];
    {
        std::lock_guard<std::mutex> lock(fftw_plan_mutex);
        if (!global_plan) {
            global_plan = fftwf_plan_dft_r2c_1d(BLOCK_SIZE, fft_in, fft_out, FFTW_MEASURE);
            assert(global_plan && "FFTW Plan creation error!");
        }
    }
    memcpy(fft_in, samples, sizeof(float) * BLOCK_SIZE);
    fftwf_execute(global_plan);
    float bin_avgs[NUM_BARS] = {0};
    for (int i = 0; i < NUM_BARS; ++i) {
        int b0 = bin_freq_to_fft_bin(freq_table[i]);
        int b1 = bin_freq_to_fft_bin(freq_table[i + 1]);
        if (b1 <= b0) b1 = b0 + 1;
        b1 = std::min(b1, BLOCK_SIZE / 2 + 1);
        float sum = 0.0f;
        int nsum = 0;
        for (int j = b0; j < b1; ++j) {
            float re = fft_out[j][0];
            float im = fft_out[j][1];
            float mag = sqrtf(re * re + im * im);
            sum += mag;
            nsum++;
        }
        bin_avgs[i] = (nsum > 0) ? sum / nsum : 0.0f;
    }
    std::lock_guard<std::mutex> lock(vis_data->mutex);
    for (int i = 0; i < NUM_BARS; ++i) {
        float mag = bin_avgs[i];
        float db = 20.0f * log10f(mag + 1e-10f);
        if (!std::isfinite(db)) db = DB_FLOOR;
        db = std::max(DB_FLOOR, db);
        db = std::min(DB_CEIL, db);
        float norm = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR);
        float maxref = (vis_data->bin_maxes[i] < 1e-10f) ? 1e-10f : vis_data->bin_maxes[i];
        float rel = mag / maxref;
        vis_data->bin_ema[i] = 0.4f * rel + 0.6f * vis_data->bin_ema[i];
        float finalval = norm * powf(std::max(0.0f, vis_data->bin_ema[i]), 0.75f);
        finalval = std::max(0.0f, std::min(1.0f, finalval));
        vis_data->target_mags[i] = finalval;
        if (finalval > vis_data->peaks[i] + 0.008f) {
            vis_data->peaks[i] = finalval;
            vis_data->peaks_vel[i] = 0.1f + 0.065f * finalval;
        }
        if (mag > vis_data->bin_maxes[i])
            vis_data->bin_maxes[i] = mag;
        vis_data->bin_maxes[i] *= 0.994f;
        vis_data->bin_maxes[i] = std::max(1e-10f, vis_data->bin_maxes[i]);
    }
}

static void interpolate_and_smooth(VisData *vis_data) {
    std::lock_guard<std::mutex> lock(vis_data->mutex);
    for (int i = 0; i < NUM_BARS; ++i) {
        vis_data->magnitudes[i] += SMOOTH_FACTOR * (vis_data->target_mags[i] - vis_data->magnitudes[i]);
        vis_data->magnitudes[i] = std::max(0.0f, vis_data->magnitudes[i]);
        if (vis_data->peaks[i] > vis_data->magnitudes[i] + 0.012f) {
            vis_data->peaks[i] -= SMOOTH_PEAK * vis_data->peaks_vel[i];
        }
        if (vis_data->peaks[i] < vis_data->magnitudes[i]) {
            vis_data->peaks[i] = vis_data->magnitudes[i];
        }
        vis_data->peaks[i] = std::max(0.0f, vis_data->peaks[i]);
    }
}

static void hsv_to_rgb(float h, float s, float v, uint8_t *r, uint8_t *g, uint8_t *b) {
    float rf, gf, bf;
    int i = static_cast<int>(floorf(h * 6.0f));
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);
    switch (i % 6) {
        case 0: rf = v; gf = t; bf = p; break;
        case 1: rf = q; gf = v; bf = p; break;
        case 2: rf = p; gf = v; bf = t; break;
        case 3: rf = p; gf = q; bf = v; break;
        case 4: rf = t; gf = p; bf = v; break;
        default: rf = v; gf = p; bf = q; break;
    }
    *r = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, rf * 255.0f)));
    *g = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, gf * 255.0f)));
    *b = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, bf * 255.0f)));
}

static void synthwave_color(float t, float v, uint8_t *r, uint8_t *g, uint8_t *b) {
    float hue;
    if (t < 0.3f) {
        hue = 0.83f + t * (0.8f / 0.3f);
    } else if (t < 0.7f) {
        hue = 0.66f - (t - 0.3f) * (0.16f / 0.4f);
    } else {
        hue = 0.5f - (t - 0.7f) * (0.1f / 0.3f);
    }
    hue = fmodf(hue + 1.0f, 1.0f);
    float saturation = 0.7f + 0.3f * v;
    float value = 0.6f + 0.4f * v;
    saturation = std::max(0.0f, std::min(1.0f, saturation));
    value = std::max(0.0f, std::min(1.0f, value));
    hsv_to_rgb(hue, saturation, value, r, g, b);
}

static int render_frame(AVFrame *frame, VisData *vis_data, int width, int height) {
    int ret = av_frame_make_writable(frame);
    if (ret < 0) {
        fprintf(stderr, "Error: Frame not writable (%s)\n", av_err2str(ret));
        return ret;
    }
    if (frame->format != AV_PIX_FMT_RGB24) {
        fprintf(stderr, "Error: render_frame expects AV_PIX_FMT_RGB24 input frame format.\n");
        return AVERROR(EINVAL);
    }
    uint8_t *data = frame->data[0];
    int linesize = frame->linesize[0];
    uint8_t bg_r = 26;
    uint8_t bg_g = 19;
    uint8_t bg_b = 41;
    for (int y = 0; y < height; y++) {
        uint8_t* row = data + y * linesize;
        for (int x = 0; x < width * 3; x += 3) {
            row[x + 0] = bg_r;
            row[x + 1] = bg_g;
            row[x + 2] = bg_b;
        }
    }
    float margin_h = width * 0.02f;
    float margin_v = height * 0.04f;
    float bar_gap = width * 0.005f;
    float available_width = width - 2 * margin_h - (NUM_BARS - 1) * bar_gap;
    float bar_w = (available_width > 0) ? available_width / (float)NUM_BARS : 0;
    float y_base = margin_v;
    float y_max_area = height - 2 * margin_v;
    float min_bar_h = 2.0f;
    std::lock_guard<std::mutex> lock(vis_data->mutex);
    for (int i = 0; i < NUM_BARS; i++) {
        if (bar_w <= 0) continue;
        float x0f = margin_h + i * (bar_w + bar_gap);
        float x1f = x0f + bar_w;
        float bar_val = vis_data->magnitudes[i];
        float peak_val = vis_data->peaks[i];
        float bar_height = bar_val * y_max_area;
        bar_height = std::max(min_bar_h, bar_height);
        float y0f = height - y_base;
        float y1f = height - (y_base + bar_height);
        float t = static_cast<float>(i) / (NUM_BARS > 1 ? (NUM_BARS - 1) : 1);
        float v = bar_val;
        uint8_t r, g, b;
        synthwave_color(t, v, &r, &g, &b);
        int x0 = static_cast<int>(roundf(x0f));
        int x1 = static_cast<int>(roundf(x1f));
        int y1 = static_cast<int>(roundf(y1f));
        int y0 = static_cast<int>(roundf(y0f));
        x0 = std::max(0, std::min(width, x0));
        x1 = std::max(x0, std::min(width, x1));
        y1 = std::max(0, std::min(height, y1));
        y0 = std::max(y1, std::min(height, y0));
        for (int y = y1; y < y0; y++) {
            if (y < 0 || y >= height) continue;
            uint8_t* row = data + y * linesize;
            for (int x = x0; x < x1; x++) {
                if (x < 0 || x >= width) continue;
                row[x * 3 + 0] = r;
                row[x * 3 + 1] = g;
                row[x * 3 + 2] = b;
            }
        }
        int highlight_h = std::max(1, static_cast<int>(bar_height * 0.1f + 2.0f));
        highlight_h = std::min(highlight_h, y0 - y1);
        if (highlight_h > 0) {
            uint8_t hr = static_cast<uint8_t>(std::min(255, static_cast<int>(r) + 60));
            uint8_t hg = static_cast<uint8_t>(std::min(255, static_cast<int>(g) + 60));
            uint8_t hb = static_cast<uint8_t>(std::min(255, static_cast<int>(b) + 60));
            int hy0 = y1;
            int hy1 = std::min(y0, y1 + highlight_h);
            int hx0 = x0;
            int hx1 = x1;
            for (int y = hy0; y < hy1; y++) {
                if (y < 0 || y >= height) continue;
                uint8_t* row = data + y * linesize;
                float factor = static_cast<float>(y - hy0) / static_cast<float>(highlight_h);
                factor = 1.0f - factor;
                uint8_t cur_r = static_cast<uint8_t>(hr * factor + r * (1.0f - factor));
                uint8_t cur_g = static_cast<uint8_t>(hg * factor + g * (1.0f - factor));
                uint8_t cur_b = static_cast<uint8_t>(hb * factor + b * (1.0f - factor));
                for (int x = hx0; x < hx1; x++) {
                    if (x < 0 || x >= width) continue;
                    row[x * 3 + 0] = cur_r;
                    row[x * 3 + 1] = cur_g;
                    row[x * 3 + 2] = cur_b;
                }
            }
        }
        float peak_min_diff = 0.015f;
        float peak_height_abs = peak_val * y_max_area;
        peak_height_abs = std::max(min_bar_h, peak_height_abs);
        if (peak_height_abs > bar_height + peak_min_diff * y_max_area && peak_val > 0.01f) {
            int peak_marker_h = 3;
            float y_peak_center = height - (y_base + peak_height_abs);
            int py0 = static_cast<int>(roundf(y_peak_center - peak_marker_h / 2.0f));
            int py1 = py0 + peak_marker_h;
            py0 = std::max(0, std::min(height, py0));
            py1 = std::max(py0, std::min(height, py1));
            uint8_t pr = static_cast<uint8_t>(std::min(255, static_cast<int>(r) + 40));
            uint8_t pg = static_cast<uint8_t>(std::min(255, static_cast<int>(g) + 40));
            uint8_t pb = static_cast<uint8_t>(std::min(255, static_cast<int>(b) + 40));
            int px_inset = std::max(1, static_cast<int>(bar_w * 0.1f));
            int px0 = std::min(width, x0 + px_inset);
            int px1 = std::max(px0, x1 - px_inset);
            for (int y = py0; y < py1; y++) {
                if (y < 0 || y >= height) continue;
                uint8_t* row = data + y * linesize;
                for (int x = px0; x < px1; x++) {
                    if (x < 0 || x >= width) continue;
                    row[x * 3 + 0] = pr;
                    row[x * 3 + 1] = pg;
                    row[x * 3 + 2] = pb;
                }
            }
        }
    }
    return 0;
}

AVFormatContext *audio_fmt_ctx = NULL;
AVCodecContext *audio_dec_ctx = NULL;
AVFormatContext *video_fmt_ctx = NULL;
AVCodecContext *video_enc_ctx = NULL;
AVCodecContext *audio_enc_ctx = NULL;
AVFrame *audio_frame = NULL;
AVFrame *rgb_frame = NULL;
AVFrame *yuv_frame = NULL;
AVPacket *audio_packet = NULL;
AVPacket *video_packet = NULL;
struct SwsContext *sws_ctx = NULL;
SwrContext *swr_enc_ctx = NULL;
SwrContext *swr_vis_ctx = NULL;

static void cleanup_resources() {
    {
        std::lock_guard<std::mutex> lock(fftw_plan_mutex);
        if (global_plan) {
            fftwf_destroy_plan(global_plan);
            global_plan = NULL;
        }
    }
    av_packet_free(&audio_packet);
    av_packet_free(&video_packet);
    av_frame_free(&audio_frame);
    av_frame_free(&rgb_frame);
    av_frame_free(&yuv_frame);
    sws_freeContext(sws_ctx);
    swr_free(&swr_enc_ctx);
    swr_free(&swr_vis_ctx);
    avcodec_free_context(&audio_dec_ctx);
    avcodec_free_context(&video_enc_ctx);
    avcodec_free_context(&audio_enc_ctx);
    if (audio_fmt_ctx) {
        avformat_close_input(&audio_fmt_ctx);
        audio_fmt_ctx = NULL;
    }
    if (video_fmt_ctx) {
        if (!(video_fmt_ctx->oformat->flags & AVFMT_NOFILE) && video_fmt_ctx->pb) {
            avio_closep(&video_fmt_ctx->pb);
        }
        avformat_free_context(video_fmt_ctx);
        video_fmt_ctx = NULL;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_audio> <output_video.mp4>\n", argv[0]);
        return 1;
    }
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    VisData vis_data = {};
    for (int i = 0; i < NUM_BARS; ++i) {
        vis_data.bin_maxes[i] = 1e-6f;
        vis_data.bin_ema[i] = 0.0f;
        vis_data.peaks[i] = 0.0f;
        vis_data.peaks_vel[i] = 0.1f;
    }
    make_freq_table();
    int ret;
    if ((ret = avformat_open_input(&audio_fmt_ctx, input_filename, NULL, NULL)) < 0) {
        fprintf(stderr, "Error opening input audio file '%s': %s\n", input_filename, av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    if ((ret = avformat_find_stream_info(audio_fmt_ctx, NULL)) < 0) {
        fprintf(stderr, "Error finding audio stream info: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    int audio_stream_index = av_find_best_stream(audio_fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
    if (audio_stream_index < 0) {
        fprintf(stderr, "Error: No audio stream found in '%s': %s\n", input_filename, av_err2str(audio_stream_index));
        cleanup_resources();
        return 1;
    }
    AVStream *audio_stream = audio_fmt_ctx->streams[audio_stream_index];
    AVCodecParameters *audio_codecpar = audio_stream->codecpar;
    const AVCodec *audio_decoder = avcodec_find_decoder(audio_codecpar->codec_id);
    if (!audio_decoder) {
        fprintf(stderr, "Error: Unsupported audio codec (ID %d)\n", audio_codecpar->codec_id);
        cleanup_resources();
        return 1;
    }
    audio_dec_ctx = avcodec_alloc_context3(audio_decoder);
    if (!audio_dec_ctx) {
        fprintf(stderr, "Error: Failed to allocate audio codec context\n");
        cleanup_resources();
        return 1;
    }
    if ((ret = avcodec_parameters_to_context(audio_dec_ctx, audio_codecpar)) < 0) {
        fprintf(stderr, "Error: Failed to copy audio codec parameters: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    if (audio_dec_ctx->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
        av_channel_layout_default(&audio_dec_ctx->ch_layout, audio_dec_ctx->ch_layout.nb_channels);
    }
    if ((ret = avcodec_open2(audio_dec_ctx, audio_decoder, NULL)) < 0) {
        fprintf(stderr, "Error opening audio decoder: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    if ((ret = avformat_alloc_output_context2(&video_fmt_ctx, NULL, NULL, output_filename)) < 0) {
        fprintf(stderr, "Error creating output context for '%s': %s\n", output_filename, av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    const AVCodec *video_encoder = avcodec_find_encoder(video_fmt_ctx->oformat->video_codec);
    if (!video_encoder) {
        video_encoder = avcodec_find_encoder_by_name("libx264");
        if (!video_encoder) {
            fprintf(stderr, "Error: Video encoder 'libx264' not found.\n");
            cleanup_resources();
            return 1;
        }
    }
    AVStream *video_stream = avformat_new_stream(video_fmt_ctx, video_encoder);
    if (!video_stream) {
        fprintf(stderr, "Error creating video stream\n");
        cleanup_resources();
        return 1;
    }
    video_enc_ctx = avcodec_alloc_context3(video_encoder);
    if (!video_enc_ctx) {
        fprintf(stderr, "Error: Failed to allocate video encoder context\n");
        cleanup_resources();
        return 1;
    }
    video_enc_ctx->codec_id = video_fmt_ctx->oformat->video_codec;
    video_enc_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
    video_enc_ctx->width = 1024;
    video_enc_ctx->height = 520;
    video_enc_ctx->time_base = (AVRational){1, 30};
    video_enc_ctx->framerate = (AVRational){30, 1};
    video_enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    video_enc_ctx->bit_rate = 4000000;
    av_opt_set(video_enc_ctx->priv_data, "preset", "medium", 0);
    if (video_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        video_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if ((ret = avcodec_open2(video_enc_ctx, video_encoder, NULL)) < 0) {
        fprintf(stderr, "Error opening video encoder: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    if ((ret = avcodec_parameters_from_context(video_stream->codecpar, video_enc_ctx)) < 0) {
        fprintf(stderr, "Error copying video encoder parameters to stream: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    const AVCodec *audio_encoder = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!audio_encoder) {
        fprintf(stderr, "Error: AAC encoder not found\n");
        cleanup_resources();
        return 1;
    }
    AVStream *out_audio_stream = avformat_new_stream(video_fmt_ctx, audio_encoder);
    if (!out_audio_stream) {
        fprintf(stderr, "Error creating output audio stream\n");
        cleanup_resources();
        return 1;
    }
    audio_enc_ctx = avcodec_alloc_context3(audio_encoder);
    if (!audio_enc_ctx) {
        fprintf(stderr, "Error: Failed to allocate audio encoder context\n");
        cleanup_resources();
        return 1;
    }
    audio_enc_ctx->sample_rate = audio_dec_ctx->sample_rate;
    audio_enc_ctx->ch_layout = audio_dec_ctx->ch_layout;
    audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP;
    audio_enc_ctx->bit_rate = 128000;
    audio_enc_ctx->time_base = (AVRational){1, audio_enc_ctx->sample_rate};
    if (video_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        audio_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if ((ret = avcodec_open2(audio_enc_ctx, audio_encoder, NULL)) < 0) {
        fprintf(stderr, "Error opening audio encoder: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    if ((ret = avcodec_parameters_from_context(out_audio_stream->codecpar, audio_enc_ctx)) < 0) {
        fprintf(stderr, "Error copying audio encoder parameters to stream: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    if (!(video_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if ((ret = avio_open(&video_fmt_ctx->pb, output_filename, AVIO_FLAG_WRITE)) < 0) {
            fprintf(stderr, "Error opening output file '%s': %s\n", output_filename, av_err2str(ret));
            cleanup_resources();
            return 1;
        }
    }
    if ((ret = avformat_write_header(video_fmt_ctx, NULL)) < 0) {
        fprintf(stderr, "Error writing output header: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }

    // Resampler for visualization (always mono 44.1K FLTP)
    AVChannelLayout mono_layout;
    av_channel_layout_default(&mono_layout, 1);
    if (swr_alloc_set_opts2(
        &swr_vis_ctx,
        &mono_layout,
        AV_SAMPLE_FMT_FLTP,
        SAMPLE_RATE,
        &audio_dec_ctx->ch_layout,
        audio_dec_ctx->sample_fmt,
        audio_dec_ctx->sample_rate,
        0, NULL) < 0) {
	    swr_vis_ctx = NULL;
    }
    if (!swr_vis_ctx || swr_init(swr_vis_ctx) < 0) {
        fprintf(stderr, "Error initializing visualization resampler.\n");
        cleanup_resources();
        return 1;
    }
    // Resampler for encoder, if needed
    if (audio_dec_ctx->sample_fmt != audio_enc_ctx->sample_fmt ||
        av_channel_layout_compare(&audio_dec_ctx->ch_layout, &audio_enc_ctx->ch_layout) != 0 ||
        audio_dec_ctx->sample_rate != audio_enc_ctx->sample_rate) {
        if (swr_alloc_set_opts2(
            &swr_enc_ctx,
            &audio_enc_ctx->ch_layout,
            audio_enc_ctx->sample_fmt,
            audio_enc_ctx->sample_rate,
            &audio_dec_ctx->ch_layout,
            audio_dec_ctx->sample_fmt,
            audio_dec_ctx->sample_rate,
            0, NULL) < 0) {
		swr_enc_ctx = NULL;
	}
        if (!swr_enc_ctx || swr_init(swr_enc_ctx) < 0) {
            fprintf(stderr, "Error initializing encoder resampler.\n");
            cleanup_resources();
            return 1;
        }
    }
    audio_packet = av_packet_alloc();
    audio_frame = av_frame_alloc();
    rgb_frame = av_frame_alloc();
    yuv_frame = av_frame_alloc();
    if (!audio_packet || !audio_frame || !rgb_frame || !yuv_frame) {
        fprintf(stderr, "Error allocating packet/frames\n");
        cleanup_resources();
        return 1;
    }
    rgb_frame->format = AV_PIX_FMT_RGB24;
    rgb_frame->width = video_enc_ctx->width;
    rgb_frame->height = video_enc_ctx->height;
    if ((ret = av_frame_get_buffer(rgb_frame, 0)) < 0) {
        fprintf(stderr, "Error allocating RGB frame buffer: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    yuv_frame->format = video_enc_ctx->pix_fmt;
    yuv_frame->width = video_enc_ctx->width;
    yuv_frame->height = video_enc_ctx->height;
    if ((ret = av_frame_get_buffer(yuv_frame, 0)) < 0) {
        fprintf(stderr, "Error allocating YUV frame buffer: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }
    sws_ctx = sws_getContext(
        video_enc_ctx->width, video_enc_ctx->height, AV_PIX_FMT_RGB24,
        video_enc_ctx->width, video_enc_ctx->height, video_enc_ctx->pix_fmt,
        SWS_BILINEAR, NULL, NULL, NULL);
    if (!sws_ctx) {
        fprintf(stderr, "Error creating SwsContext for color conversion\n");
        cleanup_resources();
        return 1;
    }

    float audio_vis_buffer[SAMPLES_PER_FRAME];
    int audio_vis_buffer_pos = 0;
    int64_t video_pts = 0;
    int64_t audio_pts = 0;

    while (av_read_frame(audio_fmt_ctx, audio_packet) >= 0) {
        if (audio_packet->stream_index == audio_stream_index) {
            ret = avcodec_send_packet(audio_dec_ctx, audio_packet);
            if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) break;
            while (ret >= 0) {
                ret = avcodec_receive_frame(audio_dec_ctx, audio_frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                else if (ret < 0) goto main_loop_end;
                //--- Visualization resample to mono 44.1K ---
                int64_t src_nb = audio_frame->nb_samples;
                AVFrame *vis_frame = av_frame_alloc();
                vis_frame->ch_layout = mono_layout;
                vis_frame->sample_rate = SAMPLE_RATE;
                vis_frame->format = AV_SAMPLE_FMT_FLTP;
                vis_frame->nb_samples = av_rescale_rnd(swr_get_delay(swr_vis_ctx, audio_frame->sample_rate) + src_nb, SAMPLE_RATE, audio_frame->sample_rate, AV_ROUND_UP);
                av_frame_get_buffer(vis_frame, 0);
                int vis_samples = swr_convert(swr_vis_ctx, vis_frame->data, vis_frame->nb_samples, (const uint8_t **)audio_frame->data, src_nb);
                vis_frame->nb_samples = vis_samples;
                float *mono_data = (float*)vis_frame->data[0];
                for (int vi = 0; vi < vis_samples; ++vi) {
                    audio_vis_buffer[audio_vis_buffer_pos++] = mono_data[vi];
                    if (audio_vis_buffer_pos >= SAMPLES_PER_FRAME) {
                        int offset = 0;
                        while (offset + BLOCK_SIZE <= SAMPLES_PER_FRAME) {
                            process_fft(&vis_data, &audio_vis_buffer[offset]);
                            offset += BLOCK_SIZE;
                        }
                        interpolate_and_smooth(&vis_data);
                        if (render_frame(rgb_frame, &vis_data, video_enc_ctx->width, video_enc_ctx->height) < 0) goto main_loop_end;
                        sws_scale(sws_ctx, (const uint8_t *const *)rgb_frame->data, rgb_frame->linesize, 0, video_enc_ctx->height, yuv_frame->data, yuv_frame->linesize);
                        yuv_frame->pts = video_pts++;
                        ret = avcodec_send_frame(video_enc_ctx, yuv_frame);
                        if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) goto main_loop_end;
                        while (ret >= 0 || ret == AVERROR(EAGAIN)) {
                            AVPacket *pkt = av_packet_alloc();
                            if (!pkt) goto main_loop_end;
                            ret = avcodec_receive_packet(video_enc_ctx, pkt);
                            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { av_packet_free(&pkt); break; }
                            else if (ret < 0) { av_packet_free(&pkt); goto main_loop_end; }
                            av_packet_rescale_ts(pkt, video_enc_ctx->time_base, video_stream->time_base);
                            pkt->stream_index = video_stream->index;
                            ret = av_interleaved_write_frame(video_fmt_ctx, pkt);
                            av_packet_free(&pkt);
                            if (ret < 0) goto main_loop_end;
                        }
                        int left = SAMPLES_PER_FRAME - offset;
                        if (left > 0) memmove(audio_vis_buffer, &audio_vis_buffer[offset], sizeof(float) * left);
                        audio_vis_buffer_pos = left;
                    }
                }
                av_frame_free(&vis_frame);
                //--- Audio encoding, with resample if needed ---
                AVFrame *enc_frame = audio_frame;
                AVFrame *enc_resampled = nullptr;
                if (swr_enc_ctx) {
                    enc_resampled = av_frame_alloc();
                    enc_resampled->ch_layout = audio_enc_ctx->ch_layout;
                    enc_resampled->sample_rate = audio_enc_ctx->sample_rate;
                    enc_resampled->format = audio_enc_ctx->sample_fmt;
                    enc_resampled->nb_samples = av_rescale_rnd(swr_get_delay(swr_enc_ctx, audio_frame->sample_rate) + audio_frame->nb_samples, audio_enc_ctx->sample_rate, audio_frame->sample_rate, AV_ROUND_UP);
                    av_frame_get_buffer(enc_resampled, 0);
                    int enc_res = swr_convert(swr_enc_ctx, enc_resampled->data, enc_resampled->nb_samples, (const uint8_t **)audio_frame->data, audio_frame->nb_samples);
                    enc_resampled->nb_samples = enc_res;
                    enc_frame = enc_resampled;
                }
                enc_frame->pts = audio_pts;
                audio_pts += enc_frame->nb_samples;
                ret = avcodec_send_frame(audio_enc_ctx, enc_frame);
                if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                    if (enc_resampled) av_frame_free(&enc_resampled);
                    av_frame_unref(audio_frame);
                    continue;
                }
                while (ret >= 0 || ret == AVERROR(EAGAIN)) {
                    AVPacket *pkt = av_packet_alloc();
                    if (!pkt) {
                        if (enc_resampled) av_frame_free(&enc_resampled);
                        av_frame_unref(audio_frame);
                        goto main_loop_end;
                    }
                    ret = avcodec_receive_packet(audio_enc_ctx, pkt);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { av_packet_free(&pkt); break; }
                    else if (ret < 0) { av_packet_free(&pkt); if (enc_resampled) av_frame_free(&enc_resampled); av_frame_unref(audio_frame); goto main_loop_end; }
                    av_packet_rescale_ts(pkt, audio_enc_ctx->time_base, out_audio_stream->time_base);
                    pkt->stream_index = out_audio_stream->index;
                    int write_ret = av_interleaved_write_frame(video_fmt_ctx, pkt);
                    av_packet_free(&pkt);
                    if (write_ret < 0) { if (enc_resampled) av_frame_free(&enc_resampled); av_frame_unref(audio_frame); goto main_loop_end; }
                }
                if (enc_resampled) av_frame_free(&enc_resampled);
                av_frame_unref(audio_frame);
            }
        }
        av_packet_unref(audio_packet);
    }
main_loop_end:
    // Flush video
    ret = avcodec_send_frame(video_enc_ctx, NULL);
    while (1) {
        AVPacket *pkt = av_packet_alloc();
        if (!pkt) break;
        ret = avcodec_receive_packet(video_enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { av_packet_free(&pkt); break; }
        else if (ret < 0) { av_packet_free(&pkt); break; }
        av_packet_rescale_ts(pkt, video_enc_ctx->time_base, video_stream->time_base);
        pkt->stream_index = video_stream->index;
        av_interleaved_write_frame(video_fmt_ctx, pkt);
        av_packet_free(&pkt);
    }
    // Flush audio
    ret = avcodec_send_frame(audio_enc_ctx, NULL);
    while (1) {
        AVPacket *pkt = av_packet_alloc();
        if (!pkt) break;
        ret = avcodec_receive_packet(audio_enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { av_packet_free(&pkt); break; }
        else if (ret < 0) { av_packet_free(&pkt); break; }
        av_packet_rescale_ts(pkt, audio_enc_ctx->time_base, out_audio_stream->time_base);
        pkt->stream_index = out_audio_stream->index;
        av_interleaved_write_frame(video_fmt_ctx, pkt);
        av_packet_free(&pkt);
    }
    if (video_fmt_ctx && !(video_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        av_write_trailer(video_fmt_ctx);
    }
    cleanup_resources();
    return 0;
}
