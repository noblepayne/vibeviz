#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fftw3.h>
#include <mutex>
#include <vector>

extern "C" {
#include <jpeglib.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#include <png.h>
}

#define SAMPLE_RATE 44100
#define SAMPLES_PER_FRAME (SAMPLE_RATE / 45)
#define BLOCK_SIZE 1024
#define NUM_BARS 36
#define SMOOTH_FACTOR 0.45f
#define SMOOTH_PEAK 0.19f
#define DB_FLOOR -65.0f
#define DB_CEIL 0.0f
#define AUDIO_CLAMP 0.98f
#define MAX_AUDIO_JUMP 2.2f

enum Theme {
  THEME_SYNTHWAVE,
  THEME_NATURE, 
  THEME_BLUE_ICE
};

static Theme current_theme = THEME_SYNTHWAVE;

struct VisData {
  float magnitudes[NUM_BARS];
  float target_mags[NUM_BARS];
  float peaks[NUM_BARS];
  float peaks_vel[NUM_BARS];
  float bin_maxes[NUM_BARS];
  float bin_ema[NUM_BARS];
  float pre_smooth[NUM_BARS];
  float silence_decay[NUM_BARS];
  std::mutex mutex;
};

static float freq_table[NUM_BARS + 1];
static fftwf_plan global_plan = NULL;
static std::mutex fftw_plan_mutex;

static void make_freq_table() {
  const float min_hz = 40.0f, max_hz = 18000.0f;
  for (int i = 0; i <= NUM_BARS; ++i) {
    float t = (float)i / (float)(NUM_BARS);
    freq_table[i] = min_hz * powf(max_hz / min_hz, t);
  }
}

static int bin_freq_to_fft_bin(float freq) {
  int bin = static_cast<int>(roundf(freq * BLOCK_SIZE / SAMPLE_RATE));
  if (bin < 0)
    bin = 0;
  if (bin > BLOCK_SIZE / 2)
    bin = BLOCK_SIZE / 2;
  return bin;
}

static void process_fft(VisData *vis_data, float *samples) {
  static float fft_in[BLOCK_SIZE];
  static fftwf_complex fft_out[BLOCK_SIZE / 2 + 1];
  {
    std::lock_guard<std::mutex> lock(fftw_plan_mutex);
    if (!global_plan) {
      global_plan =
          fftwf_plan_dft_r2c_1d(BLOCK_SIZE, fft_in, fft_out, FFTW_MEASURE);
      assert(global_plan && "FFTW Plan creation error!");
    }
  }
  float prev_max = 0.0f;
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    float v = samples[i];
    v = std::max(-AUDIO_CLAMP, std::min(AUDIO_CLAMP, v));
    fft_in[i] = v;
    prev_max = std::max(prev_max, fabsf(v));
  }
  fftwf_execute(global_plan);
  float bin_avgs[NUM_BARS] = {0};
  for (int i = 0; i < NUM_BARS; ++i) {
    int b0 = bin_freq_to_fft_bin(freq_table[i]);
    int b1 = bin_freq_to_fft_bin(freq_table[i + 1]);
    if (b1 <= b0)
      b1 = b0 + 1;
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
  // Detect silence or spike.
  bool silent = (prev_max < 1e-5f);
  for (int i = 0; i < NUM_BARS; ++i) {
    float mag = bin_avgs[i];
    if (silent)
      mag = 0.0f;
    // Spike clamp: limit maximum jump for each bin.
    float &oldval = vis_data->pre_smooth[i];
    // TODO: good enough?
    // prevent bins from sticking at zero after prolonged silence
    mag = std::max(1e-4f, mag);
    if (mag > oldval * MAX_AUDIO_JUMP)
      mag = oldval * MAX_AUDIO_JUMP;
    if (mag < oldval / MAX_AUDIO_JUMP)
      mag = oldval / MAX_AUDIO_JUMP;
    vis_data->pre_smooth[i] = mag;
    float db = 20.0f * log10f(mag + 1e-10f);
    if (!std::isfinite(db))
      db = DB_FLOOR;
    db = std::max(DB_FLOOR, db);
    db = std::min(DB_CEIL, db);
    float norm = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR);
    float maxref =
        (vis_data->bin_maxes[i] < 1e-10f) ? 1e-10f : vis_data->bin_maxes[i];
    float rel = mag / maxref;
    vis_data->bin_ema[i] = 0.4f * rel + 0.6f * vis_data->bin_ema[i];
    float finalval = norm * powf(std::max(0.0f, vis_data->bin_ema[i]), 0.75f);
    finalval = std::max(0.0f, std::min(1.0f, finalval));
    vis_data->target_mags[i] = finalval;
    if (finalval > vis_data->peaks[i] + 0.008f) {
      vis_data->peaks[i] = finalval;
      vis_data->peaks_vel[i] = 0.09f + 0.065f * finalval;
    }
    if (mag > vis_data->bin_maxes[i])
      vis_data->bin_maxes[i] = mag;
    vis_data->bin_maxes[i] *= 0.994f;
    vis_data->bin_maxes[i] = std::max(1e-10f, vis_data->bin_maxes[i]);
    // Silence smoothing/decay
    if (silent) {
      vis_data->silence_decay[i] += 0.05f;
      vis_data->target_mags[i] *= expf(-vis_data->silence_decay[i]);
    } else {
      vis_data->silence_decay[i] *= 0.7f;
    }
  }
}

static void interpolate_and_smooth(VisData *vis_data) {
  std::lock_guard<std::mutex> lock(vis_data->mutex);
  for (int i = 0; i < NUM_BARS; ++i) {
    // Smoother approach: lowpass to target, faster on rise, slower on decay
    if (vis_data->target_mags[i] > vis_data->magnitudes[i]) {
      vis_data->magnitudes[i] +=
          (SMOOTH_FACTOR + 0.17f) *
          (vis_data->target_mags[i] - vis_data->magnitudes[i]);
    } else {
      vis_data->magnitudes[i] +=
          (SMOOTH_FACTOR * 0.37f) *
          (vis_data->target_mags[i] - vis_data->magnitudes[i]);
    }
    vis_data->magnitudes[i] = std::max(0.0f, vis_data->magnitudes[i]);
    // Peak decay with cubic curve polish
    if (vis_data->peaks[i] > vis_data->magnitudes[i] + 0.012f) {
      float powdec = powf(vis_data->peaks_vel[i], 1.3f);
      vis_data->peaks[i] -= (SMOOTH_PEAK + 0.08f) * powdec;
    }
    if (vis_data->peaks[i] < vis_data->magnitudes[i]) {
      vis_data->peaks[i] = vis_data->magnitudes[i];
    }
    vis_data->peaks[i] = std::max(0.0f, vis_data->peaks[i]);
  }
}

static void hsv_to_rgb(float h, float s, float v, uint8_t *r, uint8_t *g,
                       uint8_t *b) {
  float rf, gf, bf;
  int i = static_cast<int>(floorf(h * 6.0f));
  float f = h * 6.0f - i;
  float p = v * (1.0f - s);
  float q = v * (1.0f - f * s);
  float t = v * (1.0f - (1.0f - f) * s);
  switch (i % 6) {
  case 0:
    rf = v;
    gf = t;
    bf = p;
    break;
  case 1:
    rf = q;
    gf = v;
    bf = p;
    break;
  case 2:
    rf = p;
    gf = v;
    bf = t;
    break;
  case 3:
    rf = p;
    gf = q;
    bf = v;
    break;
  case 4:
    rf = t;
    gf = p;
    bf = v;
    break;
  default:
    rf = v;
    gf = p;
    bf = q;
    break;
  }
  *r = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, rf * 255.0f)));
  *g = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, gf * 255.0f)));
  *b = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, bf * 255.0f)));
}


static void synthwave_color(float t, float v, uint8_t *r, uint8_t *g, uint8_t *b) {
  switch (current_theme) {
    case THEME_SYNTHWAVE: {
      // Original synthwave theme
      float hue;
      if (t < 0.3f)
        hue = 0.83f + t * (0.8f / 0.3f);
      else if (t < 0.7f)
        hue = 0.66f - (t - 0.3f) * (0.16f / 0.4f);
      else
        hue = 0.5f - (t - 0.7f) * (0.1f / 0.3f);
      
      hue = fmodf(hue + 1.0f, 1.0f);
      float saturation = 0.72f + 0.23f * powf(v, 0.8f);
      float value = 0.63f + 0.33f * powf(v, 0.5f);
      saturation = std::max(0.0f, std::min(1.0f, saturation));
      value = std::max(0.0f, std::min(1.0f, value));
      
      hsv_to_rgb(hue, saturation, value, r, g, b);
      break;
    }
    
    case THEME_NATURE: {
      // Nature theme
      float base_hues[2] = {154.0f/360.0f, 23.0f/360.0f}; // Converted hex colors to HSV hues
      float base_sats[2] = {0.41f, 0.77f};
      float base_vals[2] = {0.40f, 0.88f};
      
      float hue = base_hues[0] + t*(base_hues[1]-base_hues[0]);
      float saturation = base_sats[0] + t*(base_sats[1]-base_sats[0]) + 0.15f*powf(v, 0.8f);
      float value = base_vals[0] + t*(base_vals[1]-base_vals[0]) + 0.2f*powf(v, 0.5f);
      
      hue = fmodf(hue + 1.0f, 1.0f);
      saturation = std::max(0.0f, std::min(1.0f, saturation));
      value = std::max(0.0f, std::min(1.0f, value));
      
      hsv_to_rgb(hue, saturation, value, r, g, b);
      break;
    }
    
    case THEME_BLUE_ICE: {
      // blue theme with white highlights
      float hue;
      // Blue gradient from deep blue (0.6) to cyan (0.5)
      if (t < 0.3f)
        hue = 0.63f - t * (0.06f / 0.3f);
      else if (t < 0.7f)
        hue = 0.6f - (t - 0.3f) * (0.05f / 0.4f);
      else
        hue = 0.57f - (t - 0.7f) * (0.05f / 0.3f);
      
      hue = fmodf(hue + 1.0f, 1.0f);
      
      // Adjust saturation and value for blue/white theme
      float saturation = 0.7f + 0.25f * powf(v, 0.6f);
      float value = 0.7f + 0.3f * powf(v, 0.4f);
      
      // Add white highlights for higher values
      if (v > 0.8f) {
        float white_mix = (v - 0.8f) / 0.2f;
        saturation *= (1.0f - white_mix * 0.7f);
        value = std::min(1.0f, value + white_mix * 0.3f);
      }
      
      saturation = std::max(0.0f, std::min(1.0f, saturation));
      value = std::max(0.0f, std::min(1.0f, value));
      
      hsv_to_rgb(hue, saturation, value, r, g, b);
      break;
    }
  }
}

static void get_background_color(uint8_t *r, uint8_t *g, uint8_t *b) {
  switch (current_theme) {
    case THEME_SYNTHWAVE:
      *r = 23; *g = 17; *b = 38;  // Original purple-ish
      break;
    case THEME_NATURE:
      *r = 23; *g = 17; *b = 38;  // Same as synthwave for now
      break;
    case THEME_BLUE_ICE:
      *r = 10; *g = 15; *b = 30;  // blue background
      break;
  }
}


// BGImage struct and loader
struct BGImage {
  std::vector<uint8_t> pixels;
  int width;
  int height;
  bool loaded;
  BGImage() : width(0), height(0), loaded(false) {}
};

static bool load_jpeg(const char *filename, BGImage &img) {
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    return false;
  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fp);
  if (jpeg_read_header(&cinfo, TRUE) != 1) {
    fclose(fp);
    jpeg_destroy_decompress(&cinfo);
    return false;
  }
  jpeg_start_decompress(&cinfo);
  img.width = cinfo.output_width;
  img.height = cinfo.output_height;
  int row_stride = img.width * cinfo.output_components;
  img.pixels.resize(img.width * img.height * 3);
  JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo,
                                                 JPOOL_IMAGE, row_stride, 1);
  for (int y = 0; y < img.height; y++) {
    jpeg_read_scanlines(&cinfo, buffer, 1);
    for (unsigned int x = 0; x < cinfo.output_width; x++) {
      if (cinfo.output_components == 3) {
        img.pixels[(y * img.width + x) * 3 + 0] = buffer[0][x * 3 + 0];
        img.pixels[(y * img.width + x) * 3 + 1] = buffer[0][x * 3 + 1];
        img.pixels[(y * img.width + x) * 3 + 2] = buffer[0][x * 3 + 2];
      } else if (cinfo.output_components == 1) {
        uint8_t v = buffer[0][x];
        img.pixels[(y * img.width + x) * 3 + 0] = v;
        img.pixels[(y * img.width + x) * 3 + 1] = v;
        img.pixels[(y * img.width + x) * 3 + 2] = v;
      }
    }
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(fp);
  img.loaded = true;
  return true;
}
static bool load_png(const char *filename, BGImage &img) {
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    return false;
  uint8_t sig[8];
  if (fread(sig, 1, 8, fp) != 8) {
    fclose(fp);
    return false;
  }
  if (png_sig_cmp(sig, 0, 8) != 0) {
    fclose(fp);
    return false;
  }
  png_structp png_ptr =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr) {
    fclose(fp);
    return false;
  }
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    fclose(fp);
    return false;
  }
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    return false;
  }
  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  png_read_info(png_ptr, info_ptr);
  img.width = png_get_image_width(png_ptr, info_ptr);
  img.height = png_get_image_height(png_ptr, info_ptr);
  png_byte color_type = png_get_color_type(png_ptr, info_ptr);
  png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  if (bit_depth == 16)
    png_set_strip_16(png_ptr);
  if (color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png_ptr);
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png_ptr);
  if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png_ptr);
  if (color_type == PNG_COLOR_TYPE_GRAY ||
      color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png_ptr);
  png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);
  png_read_update_info(png_ptr, info_ptr);
  int rowbytes = png_get_rowbytes(png_ptr, info_ptr);
  std::vector<png_bytep> row_pointers(img.height);
  img.pixels.resize(img.width * img.height * 3);
  std::vector<uint8_t> temp_row(rowbytes);
  for (int y = 0; y < img.height; y++)
    row_pointers[y] = &temp_row[0];
  for (int y = 0; y < img.height; y++) {
    png_read_row(png_ptr, row_pointers[y], NULL);
    uint8_t *src = row_pointers[y];
    for (int x = 0; x < img.width; x++) {
      img.pixels[(y * img.width + x) * 3 + 0] = src[x * 4 + 0];
      img.pixels[(y * img.width + x) * 3 + 1] = src[x * 4 + 1];
      img.pixels[(y * img.width + x) * 3 + 2] = src[x * 4 + 2];
    }
  }
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  fclose(fp);
  img.loaded = true;
  return true;
}
static bool load_image(const char *filename, BGImage &img) {
  const char *dot = strrchr(filename, '.');
  if (!dot)
    return false;
  if (strcasecmp(dot, ".jpg") == 0 || strcasecmp(dot, ".jpeg") == 0)
    return load_jpeg(filename, img);
  if (strcasecmp(dot, ".png") == 0)
    return load_png(filename, img);
  return false;
}
// End BGImage

static int render_frame(AVFrame *frame, VisData *vis_data, int width,
                        int height, BGImage *bgimg) {
  int ret = av_frame_make_writable(frame);
  if (ret < 0) {
    fprintf(stderr, "Error: Frame not writable (%s)\n", av_err2str(ret));
    return ret;
  }
  if (frame->format != AV_PIX_FMT_RGB24) {
    fprintf(
        stderr,
        "Error: render_frame expects AV_PIX_FMT_RGB24 input frame format.\n");
    return AVERROR(EINVAL);
  }
  uint8_t *data = frame->data[0];
  int linesize = frame->linesize[0];

  if (bgimg && bgimg->loaded && bgimg->width == width &&
      bgimg->height == height) {
    for (int y = 0; y < height; y++) {
      uint8_t *row = data + y * linesize;
      const uint8_t *src = &bgimg->pixels[y * width * 3];
      memcpy(row, src, width * 3);
    }
  } else {
    // uint8_t bg_r = 23, bg_g = 17, bg_b = 38;
    uint8_t bg_r, bg_g, bg_b;
    get_background_color(&bg_r, &bg_g, &bg_b);
    for (int y = 0; y < height; y++) {
      uint8_t *row = data + y * linesize;
      for (int x = 0; x < width * 3; x += 3) {
        row[x + 0] = bg_r;
        row[x + 1] = bg_g;
        row[x + 2] = bg_b;
      }
    }
  }

  float margin_h = width * 0.02f;
  // float margin_v = height * 0.042f; ORIGINAL
  float margin_v = height * 0.15f; // for JB graphic
  float bar_gap = width * 0.0045f;
  float available_width = width - 2 * margin_h - (NUM_BARS - 1) * bar_gap;
  float bar_w = (available_width > 0) ? available_width / (float)NUM_BARS : 0;
  float y_base = margin_v;
  float y_max_area = height - 2 * margin_v;
  float min_bar_h = 2.0f;
  std::lock_guard<std::mutex> lock(vis_data->mutex);
  for (int i = 0; i < NUM_BARS; i++) {
    if (bar_w <= 0)
      continue;
    float x0f = margin_h + i * (bar_w + bar_gap);
    float x1f = x0f + bar_w;
    float bar_val = vis_data->magnitudes[i];
    float peak_val = vis_data->peaks[i];
    bar_val = std::max(0.0f, std::min(1.0f, bar_val));
    peak_val = std::max(0.0f, std::min(1.0f, peak_val));
    // Visual polish: add bar curvature smoothing.
    float bar_height = powf(bar_val, 0.83f) * y_max_area;
    bar_height = std::max(min_bar_h, bar_height);
    float y0f = height - y_base;
    float y1f = height - (y_base + bar_height);
    float t = (float)i / (NUM_BARS > 1 ? (NUM_BARS - 1) : 1);
    float v = bar_val;
    uint8_t r, g, b;
    synthwave_color(t, v, &r, &g, &b);
    int x0 = static_cast<int>(roundf(x0f));
    int x1 = static_cast<int>(roundf(x1f));
    int y1i = static_cast<int>(roundf(y1f));
    int y0i = static_cast<int>(roundf(y0f));
    x0 = std::max(0, std::min(width, x0));
    x1 = std::max(x0, std::min(width, x1));
    y1i = std::max(0, std::min(height, y1i));
    y0i = std::max(y1i, std::min(height, y0i));
    for (int y = y1i; y < y0i; y++) {
      if (y < 0 || y >= height)
        continue;
      uint8_t *row = data + y * linesize;
      for (int x = x0; x < x1; x++) {
        if (x < 0 || x >= width)
          continue;
        row[x * 3 + 0] = r;
        row[x * 3 + 1] = g;
        row[x * 3 + 2] = b;
      }
    }
    int highlight_h = std::max(1, static_cast<int>(bar_height * 0.13f + 2.0f));
    highlight_h = std::min(highlight_h, y0i - y1i);
    if (highlight_h > 0) {
      uint8_t hr = static_cast<uint8_t>(std::min(255, (int)r + 60));
      uint8_t hg = static_cast<uint8_t>(std::min(255, (int)g + 60));
      uint8_t hb = static_cast<uint8_t>(std::min(255, (int)b + 60));
      int hy0 = y1i;
      int hy1 = std::min(y0i, y1i + highlight_h);
      int hx0 = x0;
      int hx1 = x1;
      for (int y = hy0; y < hy1; y++) {
        if (y < 0 || y >= height)
          continue;
        uint8_t *row = data + y * linesize;
        float factor = (float)(y - hy0) / (float)highlight_h;
        factor = 1.0f - (factor * factor); // smoother highlight (polish)
        uint8_t cur_r = (uint8_t)(hr * factor + r * (1.0f - factor));
        uint8_t cur_g = (uint8_t)(hg * factor + g * (1.0f - factor));
        uint8_t cur_b = (uint8_t)(hb * factor + b * (1.0f - factor));
        for (int x = hx0; x < hx1; x++) {
          if (x < 0 || x >= width)
            continue;
          row[x * 3 + 0] = cur_r;
          row[x * 3 + 1] = cur_g;
          row[x * 3 + 2] = cur_b;
        }
      }
    }
    float peak_min_diff = 0.015f;
    float peak_height_abs = powf(peak_val, 0.83f) * y_max_area;
    peak_height_abs = std::max(min_bar_h, peak_height_abs);
    if (peak_height_abs > bar_height + peak_min_diff * y_max_area &&
        peak_val > 0.01f) {
      int peak_marker_h = 3;
      float y_peak_center = height - (y_base + peak_height_abs);
      int py0 = (int)roundf(y_peak_center - peak_marker_h / 2.0f);
      int py1 = py0 + peak_marker_h;
      py0 = std::max(0, std::min(height, py0));
      py1 = std::max(py0, std::min(height, py1));
      uint8_t pr = (uint8_t)(std::min(255, (int)r + 43));
      uint8_t pg = (uint8_t)(std::min(255, (int)g + 43));
      uint8_t pb = (uint8_t)(std::min(255, (int)b + 43));
      int px_inset = std::max(1, (int)(bar_w * 0.13f));
      int px0 = std::min(width, x0 + px_inset);
      int px1 = std::max(px0, x1 - px_inset);
      for (int y = py0; y < py1; y++) {
        if (y < 0 || y >= height)
          continue;
        uint8_t *row = data + y * linesize;
        for (int x = px0; x < px1; x++) {
          if (x < 0 || x >= width)
            continue;
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

struct FloatAudioFifo {
  std::vector<std::vector<float>> channels;
  int nb_channels;
  int capacity;
  int size;
  int start;
  FloatAudioFifo() : nb_channels(0), capacity(0), size(0), start(0) {}
  void init(int chans, int initial_capacity = 16384) {
    nb_channels = chans;
    capacity = initial_capacity;
    size = 0;
    start = 0;
    channels.clear();
    channels.resize(chans);
    for (int i = 0; i < chans; ++i)
      channels[i].resize(initial_capacity);
  }
  int available() const { return size; }
  int space() const { return capacity - size; }
  void push(const float **src, int nb_samples) {
    if (nb_samples <= 0)
      return;
    if (size + nb_samples > capacity) {
      int newcap = std::max(capacity * 2, size + nb_samples);
      for (int c = 0; c < nb_channels; ++c) {
        channels[c].resize(newcap);
        if (start > 0) {
          std::rotate(channels[c].begin(), channels[c].begin() + start,
                      channels[c].begin() + start + size);
          start = 0;
        }
      }
      capacity = newcap;
    }
    for (int c = 0; c < nb_channels; ++c) {
      if (start + size + nb_samples <= capacity) {
        std::copy(src[c], src[c] + nb_samples,
                  channels[c].begin() + start + size);
      } else {
        int tail = capacity - (start + size);
        std::copy(src[c], src[c] + tail, channels[c].begin() + start + size);
        std::copy(src[c] + tail, src[c] + nb_samples, channels[c].begin());
      }
    }
    size += nb_samples;
  }
  void pop(float **dst, int nb_samples) {
    assert(nb_samples <= size);
    for (int c = 0; c < nb_channels; ++c) {
      if (start + nb_samples <= capacity) {
        std::copy(channels[c].begin() + start,
                  channels[c].begin() + start + nb_samples, dst[c]);
      } else {
        int tail = capacity - start;
        std::copy(channels[c].begin() + start, channels[c].end(), dst[c]);
        std::copy(channels[c].begin(),
                  channels[c].begin() + (nb_samples - tail), dst[c] + tail);
      }
    }
    start = (start + nb_samples) % capacity;
    size -= nb_samples;
  }
  void pop_flush(float **dst, int nb_samples) {
    assert(nb_samples <= size);
    for (int c = 0; c < nb_channels; ++c) {
      if (start + nb_samples <= capacity) {
        std::copy(channels[c].begin() + start,
                  channels[c].begin() + start + nb_samples, dst[c]);
      } else {
        int tail = capacity - start;
        std::copy(channels[c].begin() + start, channels[c].end(), dst[c]);
        std::copy(channels[c].begin(),
                  channels[c].begin() + (nb_samples - tail), dst[c] + tail);
      }
    }
    start = (start + nb_samples) % capacity;
    size -= nb_samples;
    if (size == 0)
      start = 0;
  }
};

struct SyncController {
  double audio_pts_sec;
  double next_video_pts_sec;
  int64_t video_pts;
  int samples_consumed;
  double time_base_audio;
  double time_base_video;
  double frame_duration;
  SyncController() {
    audio_pts_sec = 0.0;
    next_video_pts_sec = 0.0;
    video_pts = 0;
    samples_consumed = 0;
    time_base_audio = 1.0 / SAMPLE_RATE;
    time_base_video = 1.0 / 45.0;
    frame_duration = time_base_video;
  }
  void set_timebase_audio(double tba) { time_base_audio = tba; }
  void set_timebase_video(double tbv) {
    time_base_video = tbv;
    frame_duration = tbv;
  }
  void reset() {
    audio_pts_sec = 0.0;
    next_video_pts_sec = 0.0;
    video_pts = 0;
    samples_consumed = 0;
  }
  void update_audio_samples(int nsmp) {
    samples_consumed += nsmp;
    audio_pts_sec += nsmp * time_base_audio;
  }
  bool need_video_for_audio() {
    return audio_pts_sec >= next_video_pts_sec - 1e-6;
  }
  void advance_video_frame() {
    next_video_pts_sec += frame_duration;
    ++video_pts;
  }
  double get_video_pts_sec() const { return next_video_pts_sec; }
  int64_t get_and_bump_video_pts() { return video_pts; }
  double get_audio_pos_sec() const { return audio_pts_sec; }
};

// CLI theme parsing
static void set_theme_from_string(const char* theme_name) {
  if (!theme_name) return;
  
  if (strcasecmp(theme_name, "original") == 0 || strcasecmp(theme_name, "synthwave") == 0) {
    current_theme = THEME_SYNTHWAVE;
    fprintf(stderr, "Using theme: %s\n", theme_name);
  } else if (strcasecmp(theme_name, "launch") == 0 || strcasecmp(theme_name, "nature") == 0) {
    current_theme = THEME_NATURE;
    fprintf(stderr, "Using theme: %s\n", theme_name);
  } else if (strcasecmp(theme_name, "blue") == 0 || strcasecmp(theme_name, "ice") == 0) {
    current_theme = THEME_BLUE_ICE;
    fprintf(stderr, "Using theme: %s\n", theme_name);
  } else {
    fprintf(stderr, "UNKNOWN THEME: %s ; Using theme: original\n", theme_name);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(
        stderr,
        "Usage: %s <input_audio> <output_path> "
        "[background_image.jpg|.png] "
        "[theme]\n\nOutput path can be a file "
        "(e.g., output.mp4) or RTMP URL (e.g., rtmp://server/app/streamkey)\n"
        "Themes: original/synthwave, launch/nature, blue/ice\n",
        argv[0]);
    return 1;
  }
  const char *input_filename = argv[1];
  const char *output_filename = argv[2];
  const char *bg_name = (argc > 3) ? argv[3] : NULL;
  const char *theme_name = (argc > 4) ? argv[4] : "original";
  set_theme_from_string(theme_name);
  BGImage bgimg;
  if (bg_name) {
    if (!load_image(bg_name, bgimg)) {
      fprintf(
          stderr,
          "Warning: Failed to load background image '%s', using solid color.\n",
          bg_name);
    }
  }
  VisData vis_data = {};
  for (int i = 0; i < NUM_BARS; ++i) {
    vis_data.bin_maxes[i] = 1e-6f;
    vis_data.bin_ema[i] = 0.0f;
    vis_data.peaks[i] = 0.0f;
    vis_data.peaks_vel[i] = 0.1f;
    vis_data.magnitudes[i] = 0.0f;
    vis_data.target_mags[i] = 0.0f;
    vis_data.pre_smooth[i] = 0.0001f;
    vis_data.silence_decay[i] = 0.0f;
  }
  make_freq_table();
  int ret;
  if ((ret = avformat_open_input(&audio_fmt_ctx, input_filename, NULL, NULL)) <
      0) {
    fprintf(stderr, "Error opening input audio file '%s': %s\n", input_filename,
            av_err2str(ret));
    cleanup_resources();
    return 1;
  }
  if ((ret = avformat_find_stream_info(audio_fmt_ctx, NULL)) < 0) {
    fprintf(stderr, "Error finding audio stream info: %s\n", av_err2str(ret));
    cleanup_resources();
    return 1;
  }
  int audio_stream_index =
      av_find_best_stream(audio_fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
  if (audio_stream_index < 0) {
    fprintf(stderr, "Error: No audio stream found in '%s': %s\n",
            input_filename, av_err2str(audio_stream_index));
    cleanup_resources();
    return 1;
  }
  AVStream *audio_stream = audio_fmt_ctx->streams[audio_stream_index];
  AVCodecParameters *audio_codecpar = audio_stream->codecpar;
  const AVCodec *audio_decoder = avcodec_find_decoder(audio_codecpar->codec_id);
  if (!audio_decoder) {
    fprintf(stderr, "Error: Unsupported audio codec (ID %d)\n",
            audio_codecpar->codec_id);
    cleanup_resources();
    return 1;
  }
  audio_dec_ctx = avcodec_alloc_context3(audio_decoder);
  if (!audio_dec_ctx) {
    fprintf(stderr, "Error: Failed to allocate audio codec context\n");
    cleanup_resources();
    return 1;
  }
  if ((ret = avcodec_parameters_to_context(audio_dec_ctx, audio_codecpar)) <
      0) {
    fprintf(stderr, "Error: Failed to copy audio codec parameters: %s\n",
            av_err2str(ret));
    cleanup_resources();
    return 1;
  }
  if (audio_dec_ctx->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
    av_channel_layout_default(&audio_dec_ctx->ch_layout,
                              audio_dec_ctx->ch_layout.nb_channels);
  }
  if ((ret = avcodec_open2(audio_dec_ctx, audio_decoder, NULL)) < 0) {
    fprintf(stderr, "Error opening audio decoder: %s\n", av_err2str(ret));
    cleanup_resources();
    return 1;
  }

  // --- RTMP/file output adaptation starts here ---
  bool is_rtmp = strncmp(output_filename, "rtmp://", 7) == 0;
  const char *format_name = NULL;
  if (is_rtmp) {
    format_name = "flv";
  }
  if ((ret = avformat_alloc_output_context2(&video_fmt_ctx, NULL, format_name,
                                            output_filename)) < 0) {
    fprintf(stderr, "Error creating output context for '%s': %s\n",
            output_filename, av_err2str(ret));
    cleanup_resources();
    return 1;
  }

  // TODO: mkv and mp4
  //if (!is_rtmp && strstr(video_fmt_ctx->oformat->name, "mp4")) {
  if (!is_rtmp) {
    av_dict_set(&video_fmt_ctx->metadata, "movflags", "frag_keyframe+empty_moov+default_base_moof+faststart", 0);
    fprintf(stderr, "Enabled faststart for MP4 output\n");
  }

  if (is_rtmp) {
    av_dict_set(&video_fmt_ctx->metadata, "flush_packets", "1", 0);
    video_fmt_ctx->max_delay = 100 * 1000; // 100ms
  }

  fprintf(stderr, "Output format: %s\n", video_fmt_ctx->oformat->name);

  // TODO: DELETE?
  // if (is_rtmp) {
  //   video_fmt_ctx->oformat->flags |= AVFMT_NOFILE;
  // }

  const AVCodec *video_encoder =
      avcodec_find_encoder(video_fmt_ctx->oformat->video_codec);
  if (!video_encoder) {
    video_encoder = avcodec_find_encoder_by_name("libx264");
    if (!video_encoder) {
      fprintf(stderr, "Error: Video encoder 'libx264' not found.\n");
      cleanup_resources();
      return 1;
    }
  }
  // TODO: hack
  video_encoder = avcodec_find_encoder_by_name("libx264");
  AVStream *video_stream = avformat_new_stream(video_fmt_ctx, video_encoder);
  if (!video_stream) {
    fprintf(stderr, "Error creating video stream\n");
    cleanup_resources();
    return 1;
  }

  // Video encoder setup
  video_enc_ctx = avcodec_alloc_context3(video_encoder);
  if (!video_enc_ctx) {
    fprintf(stderr, "Error: Failed to allocate video encoder context\n");
    cleanup_resources();
    return 1;
  }
  // TODO: needed?
  // video_enc_ctx->codec_id = video_fmt_ctx->oformat->video_codec;
  video_enc_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  video_enc_ctx->width = 1280;
  video_enc_ctx->height = 720;
  video_enc_ctx->time_base = (AVRational){1, 45};
  video_enc_ctx->framerate = (AVRational){45, 1};
  video_enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  video_enc_ctx->bit_rate = 4000000;
  video_enc_ctx->gop_size = 90;    // Keyframe every 2 sec (45 fps * 2s)
  video_enc_ctx->max_b_frames = 0; // <--- KEY: no B-frames for RTMP streaming

  if (is_rtmp) {
    av_opt_set(video_enc_ctx->priv_data, "preset", "ultrafast", 0);
    av_opt_set(video_enc_ctx->priv_data, "tune", "zerolatency", 0);
    av_opt_set(video_enc_ctx->priv_data, "x264-params",
               "keyint=90:min-keyint=90:scenecut=0", 0); // enforce GOP
  } else {
    av_opt_set(video_enc_ctx->priv_data, "preset", "medium", 0);
  }
  if (video_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
    video_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
  if ((ret = avcodec_open2(video_enc_ctx, video_encoder, NULL)) < 0) {
    fprintf(stderr, "Error opening video encoder: %s\n", av_err2str(ret));
    cleanup_resources();
    return 1;
  }
  if ((ret = avcodec_parameters_from_context(video_stream->codecpar,
                                             video_enc_ctx)) < 0) {
    fprintf(stderr, "Error copying video encoder parameters to stream: %s\n",
            av_err2str(ret));
    cleanup_resources();
    return 1;
  }

  // *** START ADDED DIAGNOSTIC BLOCK ***
  fprintf(stderr,
          "DEBUG: Copied video_stream codecpar: width=%d, height=%d, "
          "format=%d, codec_id=%d\n",
          video_stream->codecpar->width, video_stream->codecpar->height,
          video_stream->codecpar->format, video_stream->codecpar->codec_id);

  // Sanity check
  if (video_stream->codecpar->width <= 0 ||
      video_stream->codecpar->height <= 0) {
    fprintf(stderr,
            "FATAL ERROR: Invalid dimensions (width=%d, height=%d) detected in "
            "video stream codecpar *after* copying from encoder context!\n",
            video_stream->codecpar->width, video_stream->codecpar->height);
    // You might want to force exit here if this happens
    cleanup_resources();
    return 1; // Or handle appropriately
  }
  // *** END ADDED DIAGNOSTIC BLOCK ***

  const AVCodec *audio_encoder = avcodec_find_encoder(AV_CODEC_ID_AAC);
  if (!audio_encoder) {
    fprintf(stderr, "Error: AAC encoder not found\n");
    cleanup_resources();
    return 1;
  }
  AVStream *out_audio_stream =
      avformat_new_stream(video_fmt_ctx, audio_encoder);
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
  if ((ret = avcodec_parameters_from_context(out_audio_stream->codecpar,
                                             audio_enc_ctx)) < 0) {
    fprintf(stderr, "Error copying audio encoder parameters to stream: %s\n",
            av_err2str(ret));
    cleanup_resources();
    return 1;
  }
  // For RTMP, we need to open the output even though AVFMT_NOFILE might be set
  // For file output, we only open if AVFMT_NOFILE is not set
  if (is_rtmp || !(video_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    fprintf(stderr, "Opening output '%s'...\n", output_filename);
    if ((ret = avio_open(&video_fmt_ctx->pb, output_filename,
                         AVIO_FLAG_WRITE)) < 0) {
      fprintf(stderr, "Error opening output file '%s': %s\n", output_filename,
              av_err2str(ret));
      cleanup_resources();
      return 1;
    }
    // For streaming, set non-blocking mode
    if (is_rtmp) {
      fprintf(stderr, "Setting non-blocking mode for streaming...\n");
      video_fmt_ctx->flags |= AVFMT_FLAG_NOBUFFER;
      video_fmt_ctx->flags |= AVFMT_FLAG_FLUSH_PACKETS;
    }
  }
  if ((ret = avformat_write_header(video_fmt_ctx, NULL)) < 0) {
    fprintf(stderr, "Error writing output header: %s\n", av_err2str(ret));
    cleanup_resources();
    return 1;
  }
  // --- RTMP/file output adaptation ENDS ---

  AVChannelLayout mono_layout;
  av_channel_layout_default(&mono_layout, 1);
  if (swr_alloc_set_opts2(&swr_vis_ctx, &mono_layout, AV_SAMPLE_FMT_FLTP,
                          SAMPLE_RATE, &audio_dec_ctx->ch_layout,
                          audio_dec_ctx->sample_fmt, audio_dec_ctx->sample_rate,
                          0, NULL) < 0) {
    swr_vis_ctx = NULL;
  }
  if (!swr_vis_ctx || swr_init(swr_vis_ctx) < 0) {
    fprintf(stderr, "Error initializing visualization resampler.\n");
    cleanup_resources();
    return 1;
  }
  if (audio_dec_ctx->sample_fmt != audio_enc_ctx->sample_fmt ||
      av_channel_layout_compare(&audio_dec_ctx->ch_layout,
                                &audio_enc_ctx->ch_layout) != 0 ||
      audio_dec_ctx->sample_rate != audio_enc_ctx->sample_rate) {
    if (swr_alloc_set_opts2(
            &swr_enc_ctx, &audio_enc_ctx->ch_layout, audio_enc_ctx->sample_fmt,
            audio_enc_ctx->sample_rate, &audio_dec_ctx->ch_layout,
            audio_dec_ctx->sample_fmt, audio_dec_ctx->sample_rate, 0,
            NULL) < 0) {
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
  sws_ctx = sws_getContext(video_enc_ctx->width, video_enc_ctx->height,
                           AV_PIX_FMT_RGB24, video_enc_ctx->width,
                           video_enc_ctx->height, video_enc_ctx->pix_fmt,
                           SWS_BILINEAR, NULL, NULL, NULL);
  if (!sws_ctx) {
    fprintf(stderr, "Error creating SwsContext for color conversion\n");
    cleanup_resources();
    return 1;
  }

  SyncController sync;
  sync.set_timebase_audio(1.0 / (double)audio_dec_ctx->sample_rate);
  double video_tb = (double)video_enc_ctx->time_base.num /
                    (double)video_enc_ctx->time_base.den;
  sync.set_timebase_video(video_tb);

  int nb_channels = audio_enc_ctx->ch_layout.nb_channels;
  int frame_size =
      audio_enc_ctx->frame_size > 0 ? audio_enc_ctx->frame_size : 1024;
  FloatAudioFifo enc_fifo;
  enc_fifo.init(nb_channels);
  AVFrame *out_frame = av_frame_alloc();
  out_frame->ch_layout = audio_enc_ctx->ch_layout;
  out_frame->sample_rate = audio_enc_ctx->sample_rate;
  out_frame->format = audio_enc_ctx->sample_fmt;
  out_frame->nb_samples = frame_size;
  av_frame_get_buffer(out_frame, 0);

  AVChannelLayout vmono_layout = mono_layout;

  std::vector<float> vis_sample_buffer;
  vis_sample_buffer.reserve(SAMPLES_PER_FRAME * 6);
  int prepad_blocks = 8;
  std::vector<float> prepad(BLOCK_SIZE * prepad_blocks, 0.0f);

  int64_t enc_pts = 0;
  int64_t video_pts = 0;

  bool audio_eof = false;

  for (int b = 0; b < prepad_blocks; ++b) {
    process_fft(&vis_data, prepad.data() + b * BLOCK_SIZE);
    interpolate_and_smooth(&vis_data);
  }

  int consecutive_silence = 0;
  int consecutive_empty = 0;

  while (!audio_eof) {
    int read_ok = av_read_frame(audio_fmt_ctx, audio_packet);
    if (read_ok < 0) {
      // graceful handling of audio EOF - pad silence
      consecutive_empty++;
      if (consecutive_empty > 10)
        break;
      std::vector<float> silence(BLOCK_SIZE, 0.0f);
      process_fft(&vis_data, silence.data());
      interpolate_and_smooth(&vis_data);
      while (sync.need_video_for_audio()) {
        if (render_frame(rgb_frame, &vis_data, video_enc_ctx->width,
                         video_enc_ctx->height,
                         (bgimg.loaded ? &bgimg : NULL)) < 0)
          goto main_loop_end;
        sws_scale(sws_ctx, (const uint8_t *const *)rgb_frame->data,
                  rgb_frame->linesize, 0, video_enc_ctx->height,
                  yuv_frame->data, yuv_frame->linesize);
        yuv_frame->pts = video_pts;
        yuv_frame->pict_type = AV_PICTURE_TYPE_NONE;
        video_pts++;
        int send_ret = avcodec_send_frame(video_enc_ctx, yuv_frame);
        while (send_ret >= 0 || send_ret == AVERROR(EAGAIN)) {
          AVPacket *pkt = av_packet_alloc();
          if (!pkt)
            break;
          int recv_ret = avcodec_receive_packet(video_enc_ctx, pkt);
          if (recv_ret == AVERROR(EAGAIN) || recv_ret == AVERROR_EOF) {
            av_packet_free(&pkt);
            break;
          } else if (recv_ret < 0) {
            av_packet_free(&pkt);
            break;
          }
          av_packet_rescale_ts(pkt, video_enc_ctx->time_base,
                               video_fmt_ctx->streams[0]->time_base);
          pkt->stream_index = video_fmt_ctx->streams[0]->index;
          av_interleaved_write_frame(video_fmt_ctx, pkt);
          av_packet_free(&pkt);
        }
        sync.advance_video_frame();
      }
      continue;
    }
    consecutive_empty = 0;
    if (audio_packet->stream_index == audio_stream_index) {
      ret = avcodec_send_packet(audio_dec_ctx, audio_packet);
      if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF)
        break;
      while (ret >= 0) {
        ret = avcodec_receive_frame(audio_dec_ctx, audio_frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
          audio_eof = (ret == AVERROR_EOF);
          break;
        } else if (ret < 0)
          goto main_loop_end;

        int src_nb = audio_frame->nb_samples;
        AVFrame *vis_frame = av_frame_alloc();
        vis_frame->ch_layout = vmono_layout;
        vis_frame->sample_rate = SAMPLE_RATE;
        vis_frame->format = AV_SAMPLE_FMT_FLTP;
        vis_frame->nb_samples = av_rescale_rnd(
            swr_get_delay(swr_vis_ctx, audio_frame->sample_rate) + src_nb,
            SAMPLE_RATE, audio_frame->sample_rate, AV_ROUND_UP);
        av_frame_get_buffer(vis_frame, 0);
        int vis_samples =
            swr_convert(swr_vis_ctx, vis_frame->data, vis_frame->nb_samples,
                        (const uint8_t **)audio_frame->data, src_nb);
        vis_frame->nb_samples = vis_samples;
        float *mono_data = (float *)vis_frame->data[0];

        vis_sample_buffer.insert(vis_sample_buffer.end(), mono_data,
                                 mono_data + vis_samples);
        av_frame_free(&vis_frame);

        int vis_start = 0;
        while (vis_sample_buffer.size() - vis_start >= BLOCK_SIZE) {
          process_fft(&vis_data, vis_sample_buffer.data() + vis_start);
          vis_start += BLOCK_SIZE;
        }
        vis_sample_buffer.erase(vis_sample_buffer.begin(),
                                vis_sample_buffer.begin() + vis_start);
        interpolate_and_smooth(&vis_data);

        sync.update_audio_samples(src_nb);

        while (sync.need_video_for_audio()) {
          if (render_frame(rgb_frame, &vis_data, video_enc_ctx->width,
                           video_enc_ctx->height,
                           (bgimg.loaded ? &bgimg : NULL)) < 0)
            goto main_loop_end;
          sws_scale(sws_ctx, (const uint8_t *const *)rgb_frame->data,
                    rgb_frame->linesize, 0, video_enc_ctx->height,
                    yuv_frame->data, yuv_frame->linesize);
          yuv_frame->pts = video_pts;
          video_pts++;
          int send_ret = avcodec_send_frame(video_enc_ctx, yuv_frame);
          while (send_ret >= 0 || send_ret == AVERROR(EAGAIN)) {
            AVPacket *pkt = av_packet_alloc();
            if (!pkt)
              break;
            int recv_ret = avcodec_receive_packet(video_enc_ctx, pkt);
            if (recv_ret == AVERROR(EAGAIN) || recv_ret == AVERROR_EOF) {
              av_packet_free(&pkt);
              break;
            } else if (recv_ret < 0) {
              av_packet_free(&pkt);
              break;
            }
            av_packet_rescale_ts(pkt, video_enc_ctx->time_base,
                                 video_fmt_ctx->streams[0]->time_base);
            pkt->stream_index = video_fmt_ctx->streams[0]->index;
            av_interleaved_write_frame(video_fmt_ctx, pkt);
            av_packet_free(&pkt);
          }
          sync.advance_video_frame();
        }

        AVFrame *enc_input = audio_frame;
        AVFrame *enc_resampled = NULL;
        if (swr_enc_ctx) {
          enc_resampled = av_frame_alloc();
          enc_resampled->ch_layout = audio_enc_ctx->ch_layout;
          enc_resampled->sample_rate = audio_enc_ctx->sample_rate;
          enc_resampled->format = audio_enc_ctx->sample_fmt;
          enc_resampled->nb_samples = av_rescale_rnd(
              swr_get_delay(swr_enc_ctx, audio_frame->sample_rate) +
                  audio_frame->nb_samples,
              audio_enc_ctx->sample_rate, audio_frame->sample_rate,
              AV_ROUND_UP);
          av_frame_get_buffer(enc_resampled, 0);
          int enc_res = swr_convert(
              swr_enc_ctx, enc_resampled->data, enc_resampled->nb_samples,
              (const uint8_t **)audio_frame->data, audio_frame->nb_samples);
          enc_resampled->nb_samples = enc_res;
          enc_input = enc_resampled;
        }
        const float **input_flt = (const float **)enc_input->extended_data;
        enc_fifo.push(input_flt, enc_input->nb_samples);
        if (enc_resampled)
          av_frame_free(&enc_resampled);
        av_frame_unref(audio_frame);
        while (enc_fifo.available() >= frame_size) {
          for (int c = 0; c < nb_channels; ++c)
            memset(out_frame->extended_data[c], 0, sizeof(float) * frame_size);
          float *chanptrs[AV_NUM_DATA_POINTERS];
          for (int c = 0; c < nb_channels; ++c)
            chanptrs[c] = (float *)out_frame->extended_data[c];
          enc_fifo.pop(chanptrs, frame_size);
          out_frame->nb_samples = frame_size;
          out_frame->pts = enc_pts;
          enc_pts += frame_size;
          int sendret = avcodec_send_frame(audio_enc_ctx, out_frame);
          while (sendret >= 0 || sendret == AVERROR(EAGAIN)) {
            AVPacket *pkt = av_packet_alloc();
            if (!pkt)
              break;
            int recvv = avcodec_receive_packet(audio_enc_ctx, pkt);
            if (recvv == AVERROR(EAGAIN) || recvv == AVERROR_EOF) {
              av_packet_free(&pkt);
              break;
            } else if (recvv < 0) {
              av_packet_free(&pkt);
              break;
            }
            av_packet_rescale_ts(pkt, audio_enc_ctx->time_base,
                                 out_audio_stream->time_base);
            pkt->stream_index = out_audio_stream->index;
            av_interleaved_write_frame(video_fmt_ctx, pkt);
            av_packet_free(&pkt);
          }
        }
      }
    }
    av_packet_unref(audio_packet);
  }
main_loop_end:
  if (enc_fifo.available() > 0) {
    int avail = enc_fifo.available();
    for (int c = 0; c < nb_channels; ++c)
      memset(out_frame->extended_data[c], 0, sizeof(float) * frame_size);
    float *chanptrs[AV_NUM_DATA_POINTERS];
    for (int c = 0; c < nb_channels; ++c)
      chanptrs[c] = (float *)out_frame->extended_data[c];
    enc_fifo.pop_flush(chanptrs, avail);
    out_frame->nb_samples = avail;
    out_frame->pts = enc_pts;
    enc_pts += avail;
    avcodec_send_frame(audio_enc_ctx, out_frame);
    while (1) {
      AVPacket *pkt = av_packet_alloc();
      if (!pkt)
        break;
      int ret2 = avcodec_receive_packet(audio_enc_ctx, pkt);
      if (ret2 == AVERROR(EAGAIN) || ret2 == AVERROR_EOF) {
        av_packet_free(&pkt);
        break;
      } else if (ret2 < 0) {
        av_packet_free(&pkt);
        break;
      }
      av_packet_rescale_ts(pkt, audio_enc_ctx->time_base,
                           video_fmt_ctx->streams[1]->time_base);
      pkt->stream_index = video_fmt_ctx->streams[1]->index;
      av_interleaved_write_frame(video_fmt_ctx, pkt);
      av_packet_free(&pkt);
    }
  }
  avcodec_send_frame(video_enc_ctx, NULL);
  while (1) {
    AVPacket *pkt = av_packet_alloc();
    if (!pkt)
      break;
    ret = avcodec_receive_packet(video_enc_ctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_packet_free(&pkt);
      break;
    } else if (ret < 0) {
      av_packet_free(&pkt);
      break;
    }
    av_packet_rescale_ts(pkt, video_enc_ctx->time_base,
                         video_fmt_ctx->streams[0]->time_base);
    pkt->stream_index = video_fmt_ctx->streams[0]->index;
    av_interleaved_write_frame(video_fmt_ctx, pkt);
    av_packet_free(&pkt);
  }
  avcodec_send_frame(audio_enc_ctx, NULL);
  while (1) {
    AVPacket *pkt = av_packet_alloc();
    if (!pkt)
      break;
    ret = avcodec_receive_packet(audio_enc_ctx, pkt);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_packet_free(&pkt);
      break;
    } else if (ret < 0) {
      av_packet_free(&pkt);
      break;
    }
    av_packet_rescale_ts(pkt, audio_enc_ctx->time_base,
                         video_fmt_ctx->streams[1]->time_base);
    pkt->stream_index = video_fmt_ctx->streams[1]->index;
    av_interleaved_write_frame(video_fmt_ctx, pkt);
    av_packet_free(&pkt);
  }
  if (video_fmt_ctx && !(video_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    av_write_trailer(video_fmt_ctx);
  }
  av_frame_free(&out_frame);
  cleanup_resources();
  return 0;
}
