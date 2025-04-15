#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <assert.h>
#include <fftw3.h>
#include <math.h>
#include <pipewire/pipewire.h>
#include <pipewire/stream.h>
#include <pthread.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/audio/raw.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAMPLE_RATE 44100
#define BLOCK_SIZE 1024
#define BAR_BIN_SIZE 16
#define NUM_BARS 36
#define NUM_VERT_PER_BAR 4
#define SMOOTH_FACTOR 0.33f
#define SMOOTH_PEAK 0.13f
#define DB_FLOOR -65.0f
#define DB_CEIL 0.0f

typedef struct {
  float magnitudes[NUM_BARS];
  float target_mags[NUM_BARS];
  float peaks[NUM_BARS];
  float peaks_vel[NUM_BARS];
  float bin_maxes[NUM_BARS];
  float bin_ema[NUM_BARS];
  pthread_mutex_t mutex;
} VisData;

typedef struct {
  struct pw_stream *stream;
  struct spa_hook stream_listener;
  VisData *vis_data;
  int inbuf_pos;
  float inbuf[BLOCK_SIZE];
} AudioData;

static float freq_table[NUM_BARS + 1];

static void make_freq_table() {
  const float min_hz = 40.0f, max_hz = 18000.0f;
  for (int i = 0; i <= NUM_BARS; ++i) {
    float t = (float)i / (float)NUM_BARS;
    freq_table[i] = min_hz * powf(max_hz / min_hz, t);
  }
}

static int bin_freq_to_fft_bin(float freq) {
  int bin = (int)roundf(freq * BLOCK_SIZE / SAMPLE_RATE);
  if (bin < 0)
    bin = 0;
  if (bin > BLOCK_SIZE / 2)
    bin = BLOCK_SIZE / 2;
  return bin;
}

static void process_fft(VisData *vis_data, float *samples, int n) {
  static fftwf_plan plan = NULL;
  static float fft_in[BLOCK_SIZE];
  static fftwf_complex fft_out[BLOCK_SIZE];

  if (!plan) {
    plan = fftwf_plan_dft_r2c_1d(BLOCK_SIZE, fft_in, fft_out, FFTW_MEASURE);
    assert(plan && "FFTW Plan creation error!");
  }
  memcpy(fft_in, samples, sizeof(float) * BLOCK_SIZE);
  fftwf_execute(plan);

  float bin_avgs[NUM_BARS] = {0};

  for (int i = 0; i < NUM_BARS; ++i) {
    int b0 = bin_freq_to_fft_bin(freq_table[i]);
    int b1 = bin_freq_to_fft_bin(freq_table[i + 1]);
    if (b1 <= b0)
      b1 = b0 + 1;
    b1 = b1 > BLOCK_SIZE / 2 ? BLOCK_SIZE / 2 : b1;
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

  pthread_mutex_lock(&vis_data->mutex);
  for (int i = 0; i < NUM_BARS; ++i) {
    float mag = bin_avgs[i];
    float db = 20.0f * log10f(mag + 1e-10f);
    if (!isfinite(db))
      db = DB_FLOOR;
    db = fmaxf(DB_FLOOR, db);
    db = fminf(DB_CEIL, db);
    float norm = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR);
    float maxref =
        (vis_data->bin_maxes[i] < 1e-10f) ? 1e-10f : vis_data->bin_maxes[i];
    float rel = mag / maxref;
    vis_data->bin_ema[i] = 0.4f * rel + 0.6f * vis_data->bin_ema[i];
    float finalval = norm * powf(vis_data->bin_ema[i], 0.75f);
    vis_data->target_mags[i] = finalval;

    if (finalval > vis_data->peaks[i] + 0.008f) {
      vis_data->peaks[i] = finalval;
      vis_data->peaks_vel[i] = 0.1f + 0.065f * finalval;
    }
    if (vis_data->bin_maxes[i] < mag)
      vis_data->bin_maxes[i] = mag;
    vis_data->bin_maxes[i] *= 0.994f;
  }
  pthread_mutex_unlock(&vis_data->mutex);
}

static void on_stream_process(void *data) {
  AudioData *ad = (AudioData *)data;
  struct pw_buffer *buf;
  struct spa_buffer *spa_buf;
  buf = pw_stream_dequeue_buffer(ad->stream);
  if (!buf)
    return;
  spa_buf = buf->buffer;
  void *ptr = spa_buf->datas[0].data;
  int size = spa_buf->datas[0].chunk->size / sizeof(float);
  float *samples = (float *)ptr;
  while (size > 0) {
    int to_copy = BLOCK_SIZE - ad->inbuf_pos;
    if (to_copy > size)
      to_copy = size;
    memcpy(ad->inbuf + ad->inbuf_pos, samples, to_copy * sizeof(float));
    ad->inbuf_pos += to_copy;
    samples += to_copy;
    size -= to_copy;
    if (ad->inbuf_pos == BLOCK_SIZE) {
      process_fft(ad->vis_data, ad->inbuf, BLOCK_SIZE);
      ad->inbuf_pos = 0;
    }
  }
  pw_stream_queue_buffer(ad->stream, buf);
}

static const struct pw_stream_events stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .process = on_stream_process,
};

static int init_pipewire(AudioData *ad) {
  pw_init(NULL, NULL);
  struct pw_main_loop *loop = pw_main_loop_new(NULL);
  if (!loop)
    return -1;
  struct pw_context *context =
      pw_context_new(pw_main_loop_get_loop(loop), NULL, 0);
  if (!context)
    return -1;
  struct pw_core *core = pw_context_connect(context, NULL, 0);
  if (!core)
    return -1;

  ad->stream = pw_stream_new(core, "Capture", pw_properties_new(NULL, NULL));
  if (!ad->stream)
    return -1;
  pw_stream_add_listener(ad->stream, &ad->stream_listener, &stream_events, ad);

  struct spa_audio_info_raw info = {.format = SPA_AUDIO_FORMAT_F32,
                                    .rate = SAMPLE_RATE,
                                    .channels = 1,
                                    .position = {SPA_AUDIO_CHANNEL_MONO}};
  uint8_t buffer[1024];
  struct spa_pod_builder b;
  spa_pod_builder_init(&b, buffer, sizeof(buffer));
  struct spa_pod *param =
      spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &info);

  enum pw_stream_flags flags = (enum pw_stream_flags)(
      PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS);
  int res = pw_stream_connect(ad->stream, PW_DIRECTION_INPUT, PW_ID_ANY, flags,
                              (const struct spa_pod **)&param, 1);
  if (res < 0)
    return -1;

  pthread_t thread;
  pthread_create(&thread, NULL, (void *(*)(void *))pw_main_loop_run, loop);
  pthread_detach(thread);
  return 0;
}

static void interpolate_and_smooth(VisData *vis_data) {
  pthread_mutex_lock(&vis_data->mutex);
  for (int i = 0; i < NUM_BARS; ++i) {
    vis_data->magnitudes[i] +=
        SMOOTH_FACTOR * (vis_data->target_mags[i] - vis_data->magnitudes[i]);
    if (vis_data->peaks[i] > vis_data->magnitudes[i] + 0.012f)
      vis_data->peaks[i] -= SMOOTH_PEAK * vis_data->peaks_vel[i];
    if (vis_data->peaks[i] < vis_data->magnitudes[i])
      vis_data->peaks[i] = vis_data->magnitudes[i];
  }
  pthread_mutex_unlock(&vis_data->mutex);
}

static void update_vbo(GLuint vbo, VisData *vis_data, float width,
                       float height) {
  float margin = width * 0.02f;
  float bar_gap = width * 0.005f;
  float bar_w =
      (width - margin * 2 - bar_gap * (NUM_BARS - 1)) / (float)NUM_BARS;
  float y_base = height * 0.06f;
  float y_max = height * 0.92f; // leave for effect border

  float verts[NUM_BARS * NUM_VERT_PER_BAR * 2];
  pthread_mutex_lock(&vis_data->mutex);
  for (int i = 0; i < NUM_BARS; ++i) {
    float x0 = margin + i * (bar_w + bar_gap);
    float x1 = x0 + bar_w;
    float y0 = y_base;
    float y1 = y_base + fmaxf(vis_data->magnitudes[i] * (y_max - y_base), 4.0f);
    verts[i * 8 + 0] = x0;
    verts[i * 8 + 1] = y0;
    verts[i * 8 + 2] = x1;
    verts[i * 8 + 3] = y0;
    verts[i * 8 + 4] = x1;
    verts[i * 8 + 5] = y1;
    verts[i * 8 + 6] = x0;
    verts[i * 8 + 7] = y1;
  }
  pthread_mutex_unlock(&vis_data->mutex);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);
}

static void synthwave_color(float t, float v, float *r, float *g, float *b) {
  // Synthwave pixel art palette: purple, pink, neon blue, orange
  float s = 0.68f + 0.17f * v;
  float brightness = 0.72f + 0.28f * v;
  float hue = 0.77f - 0.64f * t; // purple->blue->pinky
  if (t > 0.70f)
    hue = 0.92f - (t - 0.7f) * 0.31f; // pinkish tail
  float h6 = hue * 6.0f;
  int hi = (int)h6;
  float f = h6 - hi;
  float q = brightness * (1 - s);
  float t1 = brightness * (1 - s * f);
  float t2 = brightness * (1 - s * (1 - f));
  switch (hi % 6) {
  case 0:
    *r = brightness;
    *g = t2;
    *b = q;
    break;
  case 1:
    *r = t1;
    *g = brightness;
    *b = q;
    break;
  case 2:
    *r = q;
    *g = brightness;
    *b = t2;
    break;
  case 3:
    *r = q;
    *g = t1;
    *b = brightness;
    break;
  case 4:
    *r = t2;
    *g = q;
    *b = brightness;
    break;
  case 5:
    *r = brightness;
    *g = q;
    *b = t1;
    break;
  }
}

static void render_bars(GLuint vao, VisData *vis_data, float width,
                        float height) {
  glBindVertexArray(vao);
  pthread_mutex_lock(&vis_data->mutex);
  for (int i = 0; i < NUM_BARS; ++i) {
    float t = (float)i / (NUM_BARS - 1);
    float v = vis_data->magnitudes[i];
    float peak = vis_data->peaks[i];
    float r, g, b;
    synthwave_color(t, v, &r, &g, &b);

    // Draw main chunky bar
    glColor3f(r, g, b);
    glDrawArrays(GL_TRIANGLE_FAN, i * NUM_VERT_PER_BAR, NUM_VERT_PER_BAR);

    // Bar top highlight
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(1.0f, 0.9f, 1.0f, (0.18f + 0.17f * v));
    float margin = width * 0.02f;
    float bar_gap = width * 0.005f;
    float bar_w =
        (width - margin * 2 - bar_gap * (NUM_BARS - 1)) / (float)NUM_BARS;
    float x0 = margin + i * (bar_w + bar_gap);
    float x1 = x0 + bar_w;
    float y_base = height * 0.06f;
    float y1 = y_base + fmaxf(vis_data->magnitudes[i] *
                                  (height * 0.92f - height * 0.06f),
                              4.0f);
    float tpad = bar_w * 0.10f;
    glBegin(GL_QUADS);
    glVertex2f(x0 + tpad, y1 - 6.0f);
    glVertex2f(x1 - tpad, y1 - 6.0f);
    glVertex2f(x1 - tpad, y1);
    glVertex2f(x0 + tpad, y1);
    glEnd();

    // Neon synthwave "edge"
    glColor4f(r * 0.6f + 0.2f, g * 0.5f + 0.2f, b * 0.9f + 0.4f,
              (0.08f + 0.18f * v));
    glBegin(GL_QUADS);
    glVertex2f(x0, y_base);
    glVertex2f(x1, y_base);
    glVertex2f(x1, y_base + 3.0f);
    glVertex2f(x0, y_base + 3.0f);
    glEnd();
    glDisable(GL_BLEND);

    // Draw peak marker
    if (peak > 0.02f) {
      float ypeak =
          y_base + fmaxf(peak * (height * 0.92f - height * 0.06f), 4.0f) - 2;
      glEnable(GL_BLEND);
      glColor4f(1.3f * r, 1.3f * g, 1.3f * b, 0.18f + 0.5f * peak);
      glBegin(GL_QUADS);
      glVertex2f(x0 + bar_w * 0.14f, ypeak - 2);
      glVertex2f(x1 - bar_w * 0.14f, ypeak - 2);
      glVertex2f(x1 - bar_w * 0.14f, ypeak + 2);
      glVertex2f(x0 + bar_w * 0.14f, ypeak + 2);
      glEnd();
      glDisable(GL_BLEND);
    }
  }
  pthread_mutex_unlock(&vis_data->mutex);
}

int main() {
  VisData vis_data = {0};
  pthread_mutex_init(&vis_data.mutex, NULL);
  for (int i = 0; i < NUM_BARS; ++i) {
    vis_data.bin_maxes[i] = 1e-6f;
    vis_data.bin_ema[i] = 1.0f;
    vis_data.peaks[i] = 0.0f;
    vis_data.peaks_vel[i] = 0.11f;
  }
  make_freq_table();

  AudioData ad = {.vis_data = &vis_data, .inbuf_pos = 0};

  if (init_pipewire(&ad)) {
    fprintf(stderr, "Error: Failed to initialize PipeWire stream\n");
    return 1;
  }
  if (!glfwInit())
    return 1;
  GLFWwindow *win = glfwCreateWindow(1024, 520, "Synthwave FFT", NULL, NULL);
  if (!win) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(win);
  glfwSwapInterval(1);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  GLuint vaos[2], vbos[2];
  glGenVertexArrays(2, vaos);
  glGenBuffers(2, vbos);
  for (int i = 0; i < 2; ++i) {
    glBindVertexArray(vaos[i]);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[i]);
    glBufferData(GL_ARRAY_BUFFER,
                 NUM_BARS * NUM_VERT_PER_BAR * 2 * sizeof(float), NULL,
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)0);
  }
  int front = 0;

  while (!glfwWindowShouldClose(win)) {
    glfwPollEvents();
    int w, h;
    glfwGetFramebufferSize(win, &w, &h);

    interpolate_and_smooth(&vis_data);

    front = 1 - front;
    update_vbo(vbos[front], &vis_data, (float)w, (float)h);

    // Synthwave dark purple background
    glClearColor(0.10f, 0.073f, 0.16f, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    render_bars(vaos[front], &vis_data, (float)w, (float)h);

    glfwSwapBuffers(win);
  }

  glfwDestroyWindow(win);
  glfwTerminate();
  return 0;
}
