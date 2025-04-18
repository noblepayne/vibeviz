#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdarg> // Needed for va_list potentially used by ffmpeg headers

#include <fftw3.h>
#include <vector>
#include <mutex>
#include <algorithm>
#include <atomic> // Potentially useful if threading were expanded

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h> // For av_image_alloc, av_frame_get_buffer etc.
#include <libavutil/error.h>   // For av_err2str
// If resampling needed: #include <libswresample/swresample.h>
}


#define SAMPLE_RATE 44100
#define BLOCK_SIZE 1024
// #define BAR_BIN_SIZE 16 // Seems unused
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
// Global FFTW plan for thread safety (if needed later, but currently single-threaded processing)
// Ensure FFTW is initialized thread-safely if used across threads.
static fftwf_plan global_plan = NULL;
static std::mutex fftw_plan_mutex; // Mutex to protect plan creation


static void make_freq_table() {
    const float min_hz = 40.0f, max_hz = 18000.0f;
    for (int i = 0; i <= NUM_BARS; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(NUM_BARS);
        freq_table[i] = min_hz * powf(max_hz / min_hz, t);
    }
}

static int bin_freq_to_fft_bin(float freq) {
    int bin = static_cast<int>(roundf(freq * BLOCK_SIZE / SAMPLE_RATE)); // Use static_cast
    if (bin < 0) bin = 0;
    // The max bin index for r2c DFT is BLOCK_SIZE/2
    if (bin > BLOCK_SIZE / 2) bin = BLOCK_SIZE / 2;
    return bin;
}

static void process_fft(VisData *vis_data, float *samples) {
    // Use thread-local storage for FFT input/output arrays if multi-threading FFT execution
    // For single-threaded audio processing loop, static is fine.
    static float fft_in[BLOCK_SIZE];
    // Output size for r2c is N/2 + 1 complex numbers
    static fftwf_complex fft_out[BLOCK_SIZE / 2 + 1];

    { // Protect plan creation/access
        std::lock_guard<std::mutex> lock(fftw_plan_mutex);
        if (!global_plan) {
            // FFTW_MEASURE can take time. Consider FFTW_ESTIMATE for faster startup.
            // Or perform planning once and save/load wisdom.
            global_plan = fftwf_plan_dft_r2c_1d(BLOCK_SIZE, fft_in, fft_out, FFTW_MEASURE);
            assert(global_plan && "FFTW Plan creation error!");
            if (!global_plan) {
                 fprintf(stderr, "Error: Failed to create FFTW plan.\n");
                 // Or handle error more gracefully
                 return; // Cannot proceed without a plan
            }
        }
    }

    // Copy samples to input buffer
    // Assuming samples pointer points to exactly BLOCK_SIZE floats
    memcpy(fft_in, samples, sizeof(float) * BLOCK_SIZE);

    // Execute the plan
    fftwf_execute(global_plan); // Execute the globally stored plan

    float bin_avgs[NUM_BARS] = {0};

    for (int i = 0; i < NUM_BARS; ++i) {
        int b0 = bin_freq_to_fft_bin(freq_table[i]);
        int b1 = bin_freq_to_fft_bin(freq_table[i + 1]);
        if (b1 <= b0) b1 = b0 + 1;
        // Ensure b1 does not exceed the bounds of fft_out
        b1 = std::min(b1, BLOCK_SIZE / 2 + 1);

        float sum = 0.0f;
        int nsum = 0;
        for (int j = b0; j < b1; ++j) {
             // Check index j against fft_out bounds (although min should handle it)
             // if (j >= BLOCK_SIZE / 2 + 1) continue; // Should not happen with std::min check

            float re = fft_out[j][0];
            float im = fft_out[j][1];
            // Compute magnitude squared first to avoid sqrt in loop if possible,
            // but here magnitude is needed for averaging.
            float mag = sqrtf(re * re + im * im);
            sum += mag;
            nsum++;
        }
        bin_avgs[i] = (nsum > 0) ? sum / nsum : 0.0f;
    }

    // Update visualization data under mutex protection
    std::lock_guard<std::mutex> lock(vis_data->mutex);
    for (int i = 0; i < NUM_BARS; ++i) {
        float mag = bin_avgs[i];
        // Avoid log(0) using a small epsilon
        float db = 20.0f * log10f(mag + 1e-10f);
        if (!std::isfinite(db)) db = DB_FLOOR; // Handle potential NaN/Inf
        db = std::max(DB_FLOOR, db);           // Floor value
        db = std::min(DB_CEIL, db);            // Ceil value
        float norm = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR); // Normalize [0, 1]

        // Relative magnitude calculation for dynamic range effect
        float maxref = (vis_data->bin_maxes[i] < 1e-10f) ? 1e-10f : vis_data->bin_maxes[i];
        float rel = mag / maxref;
        vis_data->bin_ema[i] = 0.4f * rel + 0.6f * vis_data->bin_ema[i]; // Smooth relative magnitude

        // Combine normalized dB and smoothed relative magnitude
        float finalval = norm * powf(std::max(0.0f, vis_data->bin_ema[i]), 0.75f); // Ensure base >= 0
        finalval = std::max(0.0f, std::min(1.0f, finalval)); // Clamp final value
        vis_data->target_mags[i] = finalval;

        // Peak detection logic
        if (finalval > vis_data->peaks[i] + 0.008f) {
            vis_data->peaks[i] = finalval;
            vis_data->peaks_vel[i] = 0.1f + 0.065f * finalval; // Velocity based on peak height
        }

        // Update max magnitude reference (decaying)
        if (mag > vis_data->bin_maxes[i])
            vis_data->bin_maxes[i] = mag;
        vis_data->bin_maxes[i] *= 0.994f; // Decay max reference slowly
        vis_data->bin_maxes[i] = std::max(1e-10f, vis_data->bin_maxes[i]); // Prevent from becoming zero
    }
}

static void interpolate_and_smooth(VisData *vis_data) {
    std::lock_guard<std::mutex> lock(vis_data->mutex);
    for (int i = 0; i < NUM_BARS; ++i) {
        // Smooth magnitude towards target
        vis_data->magnitudes[i] += SMOOTH_FACTOR * (vis_data->target_mags[i] - vis_data->magnitudes[i]);
        vis_data->magnitudes[i] = std::max(0.0f, vis_data->magnitudes[i]); // Prevent negative magnitudes

        // Smooth peak downwards
        // Only decrease peak if it's significantly above the current magnitude
        if (vis_data->peaks[i] > vis_data->magnitudes[i] + 0.012f) {
            vis_data->peaks[i] -= SMOOTH_PEAK * vis_data->peaks_vel[i]; // Use velocity
        }
        // Peak should not fall below current magnitude
        if (vis_data->peaks[i] < vis_data->magnitudes[i]) {
            vis_data->peaks[i] = vis_data->magnitudes[i];
        }
         vis_data->peaks[i] = std::max(0.0f, vis_data->peaks[i]); // Ensure peak is non-negative
    }
}

// HSV to RGB conversion helper (simplified)
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
        default: rf = v; gf = p; bf = q; break; // case 5
    }

    *r = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, rf * 255.0f)));
    *g = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, gf * 255.0f)));
    *b = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, bf * 255.0f)));
}


static void synthwave_color(float t, float v, uint8_t *r, uint8_t *g, uint8_t *b) {
    // Map t (position, 0..1) and v (magnitude, 0..1) to synthwave colors
    // Hue cycles from magenta/pink (low freq) through blue to cyan (high freq)
    // Saturation increases slightly with magnitude
    // Value (brightness) increases with magnitude

    float hue;
     // Transition: Pink -> Purple -> Blue -> Cyan
    if (t < 0.3f) { // Pink/Magenta region
        hue = 0.83f + t * (0.8f / 0.3f); // 0.83 (magenta) towards 0.66 (blue)
    } else if (t < 0.7f) { // Blue region
         hue = 0.66f - (t - 0.3f) * (0.16f / 0.4f); // 0.66 (blue) towards 0.5 (cyan)
    } else { // Cyan/Greenish-Cyan region
        hue = 0.5f - (t - 0.7f) * (0.1f / 0.3f); // 0.5 (cyan) towards 0.4
    }
    hue = fmodf(hue + 1.0f, 1.0f); // Ensure hue is in [0, 1]

    float saturation = 0.7f + 0.3f * v; // Saturation increases with volume
    float value = 0.6f + 0.4f * v;  // Brightness increases with volume

    // Clamp saturation and value
    saturation = std::max(0.0f, std::min(1.0f, saturation));
    value = std::max(0.0f, std::min(1.0f, value));

    hsv_to_rgb(hue, saturation, value, r, g, b);
}


static int render_frame(AVFrame *frame, VisData *vis_data, int width, int height) {
    // Ensure frame data is writable
    int ret = av_frame_make_writable(frame);
    if (ret < 0) {
        fprintf(stderr, "Error: Frame not writable (%s)\n", av_err2str(ret));
        return ret;
    }

    // Assuming AV_PIX_FMT_RGB24 format for frame input to this function
    if (frame->format != AV_PIX_FMT_RGB24) {
        fprintf(stderr, "Error: render_frame expects AV_PIX_FMT_RGB24 input frame format.\n");
        return AVERROR(EINVAL); // Invalid argument
    }
    uint8_t *data = frame->data[0];
    int linesize = frame->linesize[0];

    // Clear to dark purple background
    uint8_t bg_r = 26;
    uint8_t bg_g = 19;
    uint8_t bg_b = 41;
    for (int y = 0; y < height; y++) {
        uint8_t* row = data + y * linesize;
        for (int x = 0; x < width * 3; x += 3) {
            row[x + 0] = bg_r; // R
            row[x + 1] = bg_g; // G
            row[x + 2] = bg_b; // B
        }
    }

    // Geometry calculations
    float margin_h = width * 0.02f;
    float margin_v = height * 0.04f; // Add vertical margin
    float bar_gap = width * 0.005f;
    float available_width = width - 2 * margin_h - (NUM_BARS - 1) * bar_gap;
    float bar_w = (available_width > 0) ? available_width / (float)NUM_BARS : 0; // Prevent negative width

    float y_base = margin_v; // Start bars from bottom margin
    float y_max_area = height - 2 * margin_v; // Max height area for bars
    float min_bar_h = 2.0f; // Minimum visual height for a bar

    std::lock_guard<std::mutex> lock(vis_data->mutex); // Lock data for consistent read
    for (int i = 0; i < NUM_BARS; i++) {
        if (bar_w <= 0) continue; // Skip if bars have no width

        float x0f = margin_h + i * (bar_w + bar_gap);
        float x1f = x0f + bar_w;
        float bar_val = vis_data->magnitudes[i]; // Current magnitude [0, 1]
        float peak_val = vis_data->peaks[i]; // Current peak [0, 1]

        // Calculate bar height
        float bar_height = bar_val * y_max_area;
        bar_height = std::max(min_bar_h, bar_height); // Ensure minimum height
        float y0f = height - y_base; // Bar bottom y (graphics coords)
        float y1f = height - (y_base + bar_height); // Bar top y (graphics coords)

        // Get color based on bar index (t) and magnitude (v)
        float t = static_cast<float>(i) / (NUM_BARS > 1 ? (NUM_BARS - 1) : 1);
        float v = bar_val; // Use magnitude for primary color value
        uint8_t r, g, b;
        synthwave_color(t, v, &r, &g, &b);

        // Convert float coords to int, clamping to frame boundaries
        int x0 = static_cast<int>(roundf(x0f));
        int x1 = static_cast<int>(roundf(x1f));
        int y1 = static_cast<int>(roundf(y1f)); // Top of bar
        int y0 = static_cast<int>(roundf(y0f)); // Bottom of bar

        // Clamp coordinates strictly within frame dimensions
        x0 = std::max(0, std::min(width, x0));
        x1 = std::max(x0, std::min(width, x1)); // Ensure x1 >= x0
        y1 = std::max(0, std::min(height, y1)); // Clamp top
        y0 = std::max(y1, std::min(height, y0)); // Clamp bottom, ensure y0 >= y1


        // Draw main bar rectangle
        for (int y = y1; y < y0; y++) { // Iterate from top scanline down
             if (y < 0 || y >= height) continue; // Double check y bounds
            uint8_t* row = data + y * linesize;
            for (int x = x0; x < x1; x++) {
                if (x < 0 || x >= width) continue; // Double check x bounds
                row[x * 3 + 0] = r;
                row[x * 3 + 1] = g;
                row[x * 3 + 2] = b;
            }
        }

        // Draw top highlight (brighter, slightly inset)
        // Make highlight proportional to bar width/height? Maybe fixed size is ok.
        int highlight_h = std::max(1, static_cast<int>(bar_height * 0.1f + 2.0f)); // Small highlight height
        highlight_h = std::min(highlight_h, y0 - y1); // Cannot be taller than bar
        if (highlight_h > 0) {
            uint8_t hr = static_cast<uint8_t>(std::min(255, static_cast<int>(r) + 60));
            uint8_t hg = static_cast<uint8_t>(std::min(255, static_cast<int>(g) + 60));
            uint8_t hb = static_cast<uint8_t>(std::min(255, static_cast<int>(b) + 60));
            int hy0 = y1; // Highlight starts at the very top of the bar
            int hy1 = std::min(y0, y1 + highlight_h); // Highlight ends after highlight_h pixels or at bar bottom

            // Optional: Inset highlight horizontally
            // int hx_inset = std::max(1, static_cast<int>(bar_w * 0.1f));
            // int hx0 = std::min(width, x0 + hx_inset);
            // int hx1 = std::max(hx0, x1 - hx_inset);
             int hx0 = x0; // No inset for now
             int hx1 = x1;

            for (int y = hy0; y < hy1; y++) {
                 if (y < 0 || y >= height) continue;
                uint8_t* row = data + y * linesize;
                float factor = static_cast<float>(y - hy0) / static_cast<float>(highlight_h); // Gradient factor 0..1
                factor = 1.0f - factor; // Inverse: Brightest at top (y=hy0)
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


        // Draw peak marker (if peak is distinct)
        float peak_min_diff = 0.015f; // How much higher peak needs to be than bar
        float peak_height_abs = peak_val * y_max_area;
        peak_height_abs = std::max(min_bar_h, peak_height_abs); // Min peak height matches min bar height
        if (peak_height_abs > bar_height + peak_min_diff * y_max_area && peak_val > 0.01f) {
            int peak_marker_h = 3; // Height of the marker line
            float y_peak_center = height - (y_base + peak_height_abs);
            int py0 = static_cast<int>(roundf(y_peak_center - peak_marker_h / 2.0f));
            int py1 = py0 + peak_marker_h;


            // Clamp peak marker vertically
            py0 = std::max(0, std::min(height, py0));
            py1 = std::max(py0, std::min(height, py1)); // Ensure py1 >= py0

            // Peak marker color (slightly brighter than bar color)
             uint8_t pr = static_cast<uint8_t>(std::min(255, static_cast<int>(r) + 40));
             uint8_t pg = static_cast<uint8_t>(std::min(255, static_cast<int>(g) + 40));
             uint8_t pb = static_cast<uint8_t>(std::min(255, static_cast<int>(b) + 40));


            // Optional: Inset peak marker horizontally slightly more?
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
    return 0; // Success
}

// Helper to clean up resources
// Define context pointers globally or pass them around carefully
AVFormatContext *audio_fmt_ctx = NULL;
AVCodecContext *audio_dec_ctx = NULL;
AVFormatContext *video_fmt_ctx = NULL;
AVCodecContext *video_enc_ctx = NULL;
AVFrame *audio_frame = NULL;
AVFrame *rgb_frame = NULL;
AVFrame *yuv_frame = NULL;
AVPacket *audio_packet = NULL;
AVPacket *video_packet = NULL; // Allocate when needed
struct SwsContext *sws_ctx = NULL;
// SwrContext *swr_ctx = NULL; // If resampling needed


static void cleanup_resources() {
    printf("Cleaning up resources...\n");
    // Cleanup FFTW
    {
        std::lock_guard<std::mutex> lock(fftw_plan_mutex);
        if (global_plan) {
            fftwf_destroy_plan(global_plan);
            global_plan = NULL;
        }
        // Optional: fftwf_cleanup(); // Or fftwf_cleanup_threads();
    }


    // Free FFmpeg packets and frames
    av_packet_free(&audio_packet); // Safe even if NULL
     // video_packet is freed in the loop/flush

    av_frame_free(&audio_frame); // Safe even if NULL
    av_frame_free(&rgb_frame);
    av_frame_free(&yuv_frame);

    // Free scaler/resampler contexts
    sws_freeContext(sws_ctx); // Safe even if NULL
    // swr_free(&swr_ctx); // If resampling used

    // Close codecs
    avcodec_free_context(&audio_dec_ctx); // Safe even if NULL
    avcodec_free_context(&video_enc_ctx);

    // Close input file
    if (audio_fmt_ctx) {
        avformat_close_input(&audio_fmt_ctx); // Also frees streams etc.
        audio_fmt_ctx = NULL; // Avoid double free
    }


    // Close output file
    if (video_fmt_ctx) {
        if (!(video_fmt_ctx->oformat->flags & AVFMT_NOFILE) && video_fmt_ctx->pb) {
            avio_closep(&video_fmt_ctx->pb); // Closes file if opened
        }
        avformat_free_context(video_fmt_ctx); // Frees context and streams
        video_fmt_ctx = NULL; // Avoid double free
    }
     printf("Cleanup finished.\n");
}


int main(int argc, char *argv[]) {
    // Check command line arguments
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_audio> <output_video.mp4>\n", argv[0]);
        fprintf(stderr, "Note: Requires audio file format readable by FFmpeg.\n");
        fprintf(stderr, "      Output format is MP4 (H.264/AAC - though audio is currently discarded).\n");
        return 1;
    }
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    // Initialize VisData
    VisData vis_data = {}; // Zero initialize
    for (int i = 0; i < NUM_BARS; ++i) {
        vis_data.bin_maxes[i] = 1e-6f; // Start with small max reference
        vis_data.bin_ema[i] = 0.0f; // Start EMA at 0
        vis_data.peaks[i] = 0.0f;
        vis_data.peaks_vel[i] = 0.1f; // Default starting velocity
    }
    make_freq_table(); // Calculate frequency bins

    // --- FFmpeg Initialization & Input Setup ---
    int ret; // For error checking FFmpeg calls

    // Open input audio file
    if ((ret = avformat_open_input(&audio_fmt_ctx, input_filename, NULL, NULL)) < 0) {
        fprintf(stderr, "Error opening input audio file '%s': %s\n", input_filename, av_err2str(ret));
        cleanup_resources();
        return 1;
    }

    // Retrieve stream information
    if ((ret = avformat_find_stream_info(audio_fmt_ctx, NULL)) < 0) {
        fprintf(stderr, "Error finding audio stream info: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }

    // Find the best audio stream
    // Using av_find_best_stream for simplicity
    int audio_stream_index = av_find_best_stream(audio_fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
    if (audio_stream_index < 0) {
        fprintf(stderr, "Error: No audio stream found in '%s': %s\n", input_filename, av_err2str(audio_stream_index));
        cleanup_resources();
        return 1;
    }

    AVStream *audio_stream = audio_fmt_ctx->streams[audio_stream_index];
    AVCodecParameters *audio_codecpar = audio_stream->codecpar;

    // Find the decoder for the audio stream
    const AVCodec *audio_decoder = avcodec_find_decoder(audio_codecpar->codec_id);
    if (!audio_decoder) {
        fprintf(stderr, "Error: Unsupported audio codec (ID %d)\n", audio_codecpar->codec_id);
        cleanup_resources();
        return 1;
    }

    // Allocate a codec context for the decoder
    audio_dec_ctx = avcodec_alloc_context3(audio_decoder);
    if (!audio_dec_ctx) {
         fprintf(stderr, "Error: Failed to allocate audio codec context\n");
         cleanup_resources();
         return 1;
    }

    // Copy codec parameters from input stream to codec context
    if ((ret = avcodec_parameters_to_context(audio_dec_ctx, audio_codecpar)) < 0) {
         fprintf(stderr, "Error: Failed to copy audio codec parameters: %s\n", av_err2str(ret));
         cleanup_resources();
         return 1;
    }

    // Open the decoder
    if ((ret = avcodec_open2(audio_dec_ctx, audio_decoder, NULL)) < 0) {
        fprintf(stderr, "Error opening audio decoder: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }

     // --- Check audio format and sample rate ---
     // This visualizer currently REQUIRES specific format/rate for process_fft
      //bool needs_resampling = false;
     // bool needs_format_conversion = false; // Not implemented here

      if (audio_dec_ctx->sample_rate != SAMPLE_RATE) {
         fprintf(stderr, "Warning: Audio sample rate (%d Hz) differs from expected (%d Hz).\n", audio_dec_ctx->sample_rate, SAMPLE_RATE);
         fprintf(stderr, "         Resampling is required but NOT IMPLEMENTED. Output may be incorrect.\n");
        // needs_resampling = true; // Flag it, but can't handle it yet
      }

       // Check sample format. process_fft expects float samples (interleaved or planar handled below)
       // Preferred: AV_SAMPLE_FMT_FLT (float interleaved) or AV_SAMPLE_FMT_FLTP (float planar)
       if (audio_dec_ctx->sample_fmt != AV_SAMPLE_FMT_FLT && audio_dec_ctx->sample_fmt != AV_SAMPLE_FMT_FLTP) {
           fprintf(stderr, "Warning: Audio sample format (%s) is not float.\n", av_get_sample_fmt_name(audio_dec_ctx->sample_fmt));
           fprintf(stderr, "         Format conversion is required but NOT IMPLEMENTED. Trying to proceed may crash or produce garbage.\n");
           // needs_format_conversion = true;
           // TODO: Implement SwrContext for conversion to AV_SAMPLE_FMT_FLTP
           cleanup_resources(); // For now, exit if format isn't float.
           return 1;
       }


    // --- Output Video Setup ---
    // Allocate output format context
    if ((ret = avformat_alloc_output_context2(&video_fmt_ctx, NULL, NULL, output_filename)) < 0) {
        fprintf(stderr, "Error creating output context for '%s': %s\n", output_filename, av_err2str(ret));
        cleanup_resources();
        return 1;
    }

    // Find the encoder (e.g., H.264)
    // Try default video codec for the container first
    const AVCodec *video_encoder = avcodec_find_encoder(video_fmt_ctx->oformat->video_codec);
    if (!video_encoder) {
         // Fallback to explicitly requesting libx264 if default not found or desired
         fprintf(stderr, "Warning: Default video codec for container not found. Trying libx264.\n");
         video_encoder = avcodec_find_encoder_by_name("libx264");
         if (!video_encoder) {
            fprintf(stderr, "Error: Video encoder 'libx264' not found. Check FFmpeg build.\n");
            cleanup_resources();
            return 1;
         }
         // If we manually chose encoder, set the codec ID in the format context
	 // TODO: fix?
         // video_fmt_ctx->oformat->video_codec = video_encoder->id;
    }

    // Create a new video stream in the output context
    AVStream *video_stream = avformat_new_stream(video_fmt_ctx, video_encoder);
    if (!video_stream) {
        fprintf(stderr, "Error creating video stream\n");
        cleanup_resources();
        return 1;
    }
    // video_stream->id = video_fmt_ctx->nb_streams - 1; // Set stream ID


    // Allocate codec context for the encoder
    video_enc_ctx = avcodec_alloc_context3(video_encoder);
     if (!video_enc_ctx) {
         fprintf(stderr, "Error: Failed to allocate video encoder context\n");
         cleanup_resources();
         return 1;
    }

    // Set encoder parameters
    video_enc_ctx->codec_id = video_fmt_ctx->oformat->video_codec; // Ensure codec ID is set
    video_enc_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
    video_enc_ctx->width = 1024;
    video_enc_ctx->height = 520;
    video_enc_ctx->time_base = (AVRational){1, 30}; // 30 FPS timebase
    video_enc_ctx->framerate = (AVRational){30, 1}; // 30 FPS framerate
    // Common pixel format for H.264 compatibility. SwScale will convert to this.
    video_enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    video_enc_ctx->bit_rate = 4000000; // Example bitrate (4 Mbps)
    // Other options: gop_size, max_b_frames, profile, preset etc. can be set via av_opt_set

     // Set options like preset for H.264
     av_opt_set(video_enc_ctx->priv_data, "preset", "medium", 0);
     // av_opt_set(video_enc_ctx->priv_data, "crf", "23", 0); // Constant Rate Factor (alternative to bitrate)


     // Some formats require global headers
    if (video_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        video_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // Open the video encoder
    if ((ret = avcodec_open2(video_enc_ctx, video_encoder, NULL)) < 0) {
        fprintf(stderr, "Error opening video encoder: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }

    // Copy encoder parameters to the output stream
    if ((ret = avcodec_parameters_from_context(video_stream->codecpar, video_enc_ctx)) < 0) {
         fprintf(stderr, "Error copying video encoder parameters to stream: %s\n", av_err2str(ret));
         cleanup_resources();
         return 1;
    }


    // Open the output file if needed
    if (!(video_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if ((ret = avio_open(&video_fmt_ctx->pb, output_filename, AVIO_FLAG_WRITE)) < 0) {
            fprintf(stderr, "Error opening output file '%s': %s\n", output_filename, av_err2str(ret));
            cleanup_resources();
            return 1;
        }
    }

    // Write the stream header to the output file
    if ((ret = avformat_write_header(video_fmt_ctx, NULL)) < 0) {
        fprintf(stderr, "Error writing output header: %s\n", av_err2str(ret));
        cleanup_resources();
        return 1;
    }


    // --- Prepare for Processing Loop ---
    // Allocate packet and frames
    audio_packet = av_packet_alloc();
    audio_frame = av_frame_alloc();
    rgb_frame = av_frame_alloc();
    yuv_frame = av_frame_alloc();
    if (!audio_packet || !audio_frame || !rgb_frame || !yuv_frame) {
          fprintf(stderr, "Error allocating packet/frames\n");
          cleanup_resources(); return 1;
    }

    // Configure RGB frame for rendering
    rgb_frame->format = AV_PIX_FMT_RGB24;
    rgb_frame->width = video_enc_ctx->width;
    rgb_frame->height = video_enc_ctx->height;
    if ((ret = av_frame_get_buffer(rgb_frame, 0)) < 0) { // Use align 0 for default
         fprintf(stderr, "Error allocating RGB frame buffer: %s\n", av_err2str(ret));
         cleanup_resources(); return 1;
    }

    // Configure YUV frame for encoding
    yuv_frame->format = video_enc_ctx->pix_fmt; // Should be YUV420P
    yuv_frame->width = video_enc_ctx->width;
    yuv_frame->height = video_enc_ctx->height;
    if ((ret = av_frame_get_buffer(yuv_frame, 0)) < 0) { // Use align 0 for default
         fprintf(stderr, "Error allocating YUV frame buffer: %s\n", av_err2str(ret));
         cleanup_resources(); return 1;
    }

    // Configure SWS context for RGB -> YUV conversion
    sws_ctx = sws_getContext(
        video_enc_ctx->width, video_enc_ctx->height, AV_PIX_FMT_RGB24, // Source
        video_enc_ctx->width, video_enc_ctx->height, video_enc_ctx->pix_fmt, // Destination
        SWS_BILINEAR, NULL, NULL, NULL);
    if (!sws_ctx) {
        fprintf(stderr, "Error creating SwsContext for color conversion\n");
        cleanup_resources(); return 1;
    }


    // Audio buffer for FFT processing
    // Use a std::vector for easier management if size isn't compile-time constant
    float audio_buffer[BLOCK_SIZE];
    int audio_buffer_pos = 0;
    int64_t video_pts = 0; // Frame counter for PTS


    // --- Main Processing Loop ---
    printf("Starting processing...\n");
    while (av_read_frame(audio_fmt_ctx, audio_packet) >= 0) {
        // Check if the packet belongs to the audio stream
        if (audio_packet->stream_index == audio_stream_index) {
            // Send the packet to the decoder
            ret = avcodec_send_packet(audio_dec_ctx, audio_packet);
            if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                fprintf(stderr, "Error sending audio packet to decoder: %s\n", av_err2str(ret));
                 // Decide whether to break or continue
                 break; // Exit loop on decode error
            } // Ignore EAGAIN, handle EOF below

            // Receive decoded frames
            while (ret >= 0) {
                ret = avcodec_receive_frame(audio_dec_ctx, audio_frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    // Need more packets or end of stream
                    break;
                } else if (ret < 0) {
                    fprintf(stderr, "Error receiving decoded audio frame: %s\n", av_err2str(ret));
                    // Decide whether to break or continue
                    goto main_loop_end; // Use goto for cleanup
                }

                // --- Process Audio Samples ---
                 // Check format again inside loop (though context shouldn't change)
                 float *samples_ptr = nullptr;
                 int samples_per_channel = audio_frame->nb_samples;

                 if (audio_frame->format == AV_SAMPLE_FMT_FLTP) {
                     // Planar float: data[0] has Left, data[1] has Right, etc.
                     // Simplification: Use only the first channel (Left) for FFT
                     samples_ptr = reinterpret_cast<float*>(audio_frame->data[0]);
                 } else if (audio_frame->format == AV_SAMPLE_FMT_FLT) {
                      // Interleaved float: L R L R L R ...
                      // Need modification in process_fft or here to handle stereo properly.
                      // Simplification: Treat interleaved as mono (take every Nth sample if N channels)
                      // Or just pass the buffer and process_fft takes first BLOCK_SIZE samples.
                      samples_ptr = reinterpret_cast<float*>(audio_frame->data[0]);
                      // If stereo, maybe process only left: samples_ptr = (float*)audio_frame->data[0]; step = 2;
                 } else {
                      // Should have exited earlier if format wasn't float. Safety check.
                       fprintf(stderr, "Error: Unexpected audio format (%s) during processing.\n", av_get_sample_fmt_name(static_cast<AVSampleFormat>(audio_frame->format)));
                       av_frame_unref(audio_frame);
                       continue; // Skip this frame
                 }


                 // Fill the FFT buffer
                 for (int i = 0; i < samples_per_channel; ++i) {
                      // TODO: Handle stereo properly, e.g., average channels or use only one.
                      // Currently just taking samples sequentially, might mix L/R if interleaved.
                      // For planar, this takes only channel 0.
                       if (samples_ptr) {
                           audio_buffer[audio_buffer_pos++] = samples_ptr[i];
                       }

                       // Block filled, process FFT and generate video frame
                       if (audio_buffer_pos >= BLOCK_SIZE) {
                           process_fft(&vis_data, audio_buffer); // Process the filled block
                           audio_buffer_pos = 0; // Reset buffer position

                            // Smooth visual data between FFT updates
                           interpolate_and_smooth(&vis_data);

                           // Render the visualization onto the RGB frame
                           if (render_frame(rgb_frame, &vis_data, video_enc_ctx->width, video_enc_ctx->height) < 0) {
                               fprintf(stderr, "Error rendering video frame\n");
                               goto main_loop_end; // Use goto for cleanup
                           }

                            // Convert RGB frame to YUV frame for encoding
                           sws_scale(sws_ctx, (const uint8_t *const *)rgb_frame->data,
                                     rgb_frame->linesize, 0, video_enc_ctx->height,
                                     yuv_frame->data, yuv_frame->linesize);

                            // Set PTS for the output frame
                           yuv_frame->pts = video_pts++;

                           // --- Encode Video Frame ---
                           ret = avcodec_send_frame(video_enc_ctx, yuv_frame);
                           if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                               fprintf(stderr, "Error sending video frame to encoder: %s\n", av_err2str(ret));
                               goto main_loop_end; // Use goto for cleanup
                           }

                           // Receive encoded video packets
                           while (ret >= 0 || ret == AVERROR(EAGAIN)) { // Continue while encoder might have output
                                if (!video_packet) video_packet = av_packet_alloc();
                                if (!video_packet) {fprintf(stderr, "Error alloc video packet\n"); goto main_loop_end;}

                               ret = avcodec_receive_packet(video_enc_ctx, video_packet);
                               if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                                   // Need more input or end of stream
                                   break;
                               } else if (ret < 0) {
                                   fprintf(stderr, "Error receiving encoded video packet: %s\n", av_err2str(ret));
                                    av_packet_free(&video_packet); // Free packet before cleanup jump
                                   goto main_loop_end; // Use goto for cleanup
                               }

                               // Rescale PTS from encoder timebase to stream timebase
                               av_packet_rescale_ts(video_packet, video_enc_ctx->time_base, video_stream->time_base);
                               video_packet->stream_index = video_stream->index;

                               // Write the encoded packet to the output file
                               ret = av_interleaved_write_frame(video_fmt_ctx, video_packet);
                               if (ret < 0) {
                                  fprintf(stderr, "Error writing video frame: %s\n", av_err2str(ret));
                                  // Might want to break or continue depending on error
                               }
                               av_packet_unref(video_packet); // Unref packet after writing (or error)
                           }
                            av_packet_free(&video_packet); // Free packet if allocated in loop
                       } // End if block filled
                 } // End loop through samples in audio frame

                av_frame_unref(audio_frame); // Unreference the frame before receiving next
            } // End receive frame loop
        } // End if audio packet

        av_packet_unref(audio_packet); // Unreference the input packet
    } // End read frame loop

main_loop_end: // For jumping to cleanup
    printf("Processing loop finished.\n");

    // --- Flush Decoders and Encoders ---
    printf("Flushing codecs...\n");
    // Flush audio decoder (optional, normally done by EOF)
    // avcodec_send_packet(audio_dec_ctx, NULL);
    // while (avcodec_receive_frame(audio_dec_ctx, audio_frame) == 0) { /* Process remaining */ }

    // Flush video encoder
    ret = avcodec_send_frame(video_enc_ctx, NULL); // Send NULL frame to flush
     if (ret < 0 && ret != AVERROR_EOF) {
          fprintf(stderr, "Error sending flush frame to video encoder: %s\n", av_err2str(ret));
          // Continue to cleanup anyway
     } else {
            while (1) {
                 if (!video_packet) video_packet = av_packet_alloc();
                 if (!video_packet) { fprintf(stderr, "Error alloc video packet for flush\n"); break; } // Allocation error

                ret = avcodec_receive_packet(video_enc_ctx, video_packet);
                if (ret == AVERROR(EAGAIN)) {
                     fprintf(stderr, "Encoder needs more data to flush? Should not happen after NULL frame.\n");
                     break; // Should not happen?
                } else if (ret == AVERROR_EOF) {
                     av_packet_free(&video_packet); // Free packet before breaking
                     break; // End of flush
                } else if (ret < 0) {
                    fprintf(stderr, "Error receiving flushed video packet: %s\n", av_err2str(ret));
                    av_packet_free(&video_packet); // Free packet before breaking
                    break; // Error during flush
                }

                av_packet_rescale_ts(video_packet, video_enc_ctx->time_base, video_stream->time_base);
                video_packet->stream_index = video_stream->index;

                int write_ret = av_interleaved_write_frame(video_fmt_ctx, video_packet);
                if (write_ret < 0) {
                    fprintf(stderr, "Error writing flushed video frame: %s\n", av_err2str(write_ret));
                }
                av_packet_unref(video_packet);
            }
            av_packet_free(&video_packet); // Free packet if allocated in loop
     }


    // Write the trailer to the output file
    printf("Writing output trailer...\n");
    if (video_fmt_ctx && !(video_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = av_write_trailer(video_fmt_ctx);
        if (ret < 0) {
           fprintf(stderr, "Error writing output trailer: %s\n", av_err2str(ret));
           // Report error but continue cleanup
        }
    } else {
         fprintf(stderr, "Trailer not written (no file I/O or format context invalid).\n");
    }


    // --- Cleanup ---
     cleanup_resources();


    return 0; // Success
}
