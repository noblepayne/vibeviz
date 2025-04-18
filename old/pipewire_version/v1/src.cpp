#define GL_GLEXT_PROTOTYPES

#include <pipewire/pipewire.h>
#include <pipewire/stream.h>
#include <pipewire/core.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/audio/raw.h>
#include <spa/param/format-utils.h>
#include <fftw3.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define SAMPLE_RATE 44100
#define BLOCK_SIZE 512
#define BAR_BIN_SIZE 6
#define NUM_BARS (BLOCK_SIZE/(2*BAR_BIN_SIZE))
#define NUM_VERTICES_PER_BAR 4
#define SMOOTH_FACTOR 0.35f

#define DB_FLOOR -60.0f
#define DB_CEIL 0.0f

typedef struct {
    float magnitudes[NUM_BARS];
    float target_mags[NUM_BARS];
    float audio_level;
    float target_level;
    float bin_maxes[NUM_BARS];
    float bin_ema[NUM_BARS];
    pthread_mutex_t mutex;
} VisData;

typedef struct {
    struct pw_stream *stream;
    struct spa_hook stream_listener;
    VisData *vis_data;
    fftwf_plan fft_plan;
    float inbuf[BLOCK_SIZE];
    fftwf_complex outbuf[BLOCK_SIZE];
    int inbuf_pos;
} AudioData;

static float freq_table[NUM_BARS+1];

static void make_freq_table() {
    for (int i=0; i<=NUM_BARS; ++i) {
        float min_hz = 40.0f, max_hz = 20000.0f;
        float t = (float)i/(float)NUM_BARS;
        float hz = min_hz * powf(max_hz/min_hz, t);
        freq_table[i] = hz;
    }
}

static int bin_freq_to_fft_bin(float freq) {
    return (int)(freq * BLOCK_SIZE / SAMPLE_RATE);
}

static void process_fft(VisData *vis_data, float *samples, int n) {
    static fftwf_plan plan = NULL;
    static float fft_in[BLOCK_SIZE];
    static fftwf_complex fft_out[BLOCK_SIZE];
    if (plan == NULL) {
        plan = fftwf_plan_dft_r2c_1d(BLOCK_SIZE, fft_in, fft_out, FFTW_MEASURE);
    }
    memcpy(fft_in, samples, sizeof(float) * BLOCK_SIZE);
    fftwf_execute(plan);

    float bin_avgs[NUM_BARS] = {0};
    for (int i = 0; i < NUM_BARS; ++i) {
        int b0 = bin_freq_to_fft_bin(freq_table[i]);
        int b1 = bin_freq_to_fft_bin(freq_table[i+1]);
        b1 = b1 > BLOCK_SIZE/2 ? BLOCK_SIZE/2 : b1;
        if (b1 <= b0) b1 = b0+1;
        float sum = 0.0f;
        int nsum = 0;
        for (int j = b0; j < b1; ++j) {
            float re = fft_out[j][0];
            float im = fft_out[j][1];
            float mag = sqrtf(re*re + im*im);
            sum += mag;
            nsum++;
        }
        if (nsum > 0)
            bin_avgs[i] = sum / nsum;
        else
            bin_avgs[i] = 0.0f;
    }

    for (int i = 0; i < NUM_BARS; ++i) {
        float mag = bin_avgs[i];
        float db = 20.0f * log10f(mag + 1e-10f);
        if (!isfinite(db)) db = DB_FLOOR;
        db = fmaxf(DB_FLOOR, db);
        db = fminf(DB_CEIL, db);
        float norm = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR);

        if (vis_data->bin_maxes[i] < mag) vis_data->bin_maxes[i] = mag;
    }

    pthread_mutex_lock(&vis_data->mutex);
    for (int i = 0; i < NUM_BARS; ++i) {
        float mag = bin_avgs[i];
        float db = 20.0f * log10f(mag + 1e-10f);
        if (!isfinite(db)) db = DB_FLOOR;
        db = fmaxf(DB_FLOOR, db);
        db = fminf(DB_CEIL, db);
        float norm = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR);

        // Adaptive normalization: update max with a slow falloff
        float maxref = vis_data->bin_maxes[i];
        if (maxref < 1e-10f) maxref = 1e-10f;
        float rel = mag / maxref;
        vis_data->bin_ema[i] = 0.6f*vis_data->bin_ema[i] + 0.4f*rel;

        float finalval = norm * powf(vis_data->bin_ema[i],0.6f);

        vis_data->target_mags[i] = finalval;
        vis_data->bin_maxes[i] *= 0.995f; // slow decay
    }

    float sumsq = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; ++i) sumsq += samples[i] * samples[i];
    vis_data->target_level = sqrtf(sumsq / BLOCK_SIZE);

    pthread_mutex_unlock(&vis_data->mutex);
}

static void on_stream_process(void *data) {
    AudioData *ad = (AudioData*)data;
    struct pw_buffer *buf;
    struct spa_buffer *spa_buf;
    buf = pw_stream_dequeue_buffer(ad->stream);
    if (!buf) return;
    spa_buf = buf->buffer;
    void *ptr = spa_buf->datas[0].data;
    int size = spa_buf->datas[0].chunk->size / sizeof(float);
    float *samples = (float*)ptr;
    while (size > 0) {
        int to_copy = BLOCK_SIZE - ad->inbuf_pos;
        if (to_copy > size) to_copy = size;
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
    struct pw_properties *props;
    pw_init(NULL, NULL);
    struct pw_main_loop *loop = pw_main_loop_new(NULL);
    if (!loop) return -1;
    struct pw_context *context = pw_context_new(pw_main_loop_get_loop(loop), NULL, 0);
    if (!context) return -1;

    struct pw_core *core = pw_context_connect(context, NULL, 0);
    if (!core) return -1;
    ad->stream = pw_stream_new(core, "Capture", pw_properties_new(NULL, NULL));
    if (!ad->stream) return -1;
    pw_stream_add_listener(ad->stream, &ad->stream_listener, &stream_events, ad);

    props = pw_properties_new(
        PW_KEY_MEDIA_TYPE, "Audio",
        PW_KEY_MEDIA_CATEGORY, "Capture",
        PW_KEY_MEDIA_ROLE, "Music",
        NULL
    );

    struct spa_audio_info_raw info = {
        .format = SPA_AUDIO_FORMAT_F32,
        .rate = SAMPLE_RATE,
        .channels = 1,
        .position = { SPA_AUDIO_CHANNEL_MONO }
    };

    struct spa_pod_builder b;
    uint8_t buffer[1024];
    struct spa_pod *param;

    spa_pod_builder_init(&b, buffer, sizeof(buffer));
    param = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &info);

    enum pw_stream_flags flags = (enum pw_stream_flags)(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS);

    int res = pw_stream_connect(ad->stream, PW_DIRECTION_INPUT, PW_ID_ANY,
        flags,
        (const struct spa_pod**)&param, 1);
    if (res < 0) return -1;

    pthread_t thread;
    pthread_create(&thread, NULL, (void*(*)(void*))pw_main_loop_run, loop);
    pthread_detach(thread);
    return 0;
}

static void interpolate_and_smooth(VisData *vis_data) {
    pthread_mutex_lock(&vis_data->mutex);
    for (int i = 0; i < NUM_BARS; ++i) {
        vis_data->magnitudes[i] += SMOOTH_FACTOR * (vis_data->target_mags[i] - vis_data->magnitudes[i]);
    }
    vis_data->audio_level += SMOOTH_FACTOR * (vis_data->target_level - vis_data->audio_level);
    pthread_mutex_unlock(&vis_data->mutex);
}

static void update_vbo(GLuint vbo, VisData *vis_data, float width, float height) {
    float gap = width * 0.01f;
    float bar_w = (width - gap * (NUM_BARS + 1)) / NUM_BARS;
    float verts[NUM_BARS * NUM_VERTICES_PER_BAR * 2];
    pthread_mutex_lock(&vis_data->mutex);
    for (int i = 0; i < NUM_BARS; ++i) {
        float x0 = gap + i * (bar_w + gap);
        float x1 = x0 + bar_w;
        float y0 = 0.0f;
        float y1 = vis_data->magnitudes[i] * height * 0.92f + height * 0.06f;
        y1 = fmaxf(y1, y0 + 2);
        verts[i*8 + 0] = x0; verts[i*8 + 1] = y0;
        verts[i*8 + 2] = x1; verts[i*8 + 3] = y0;
        verts[i*8 + 4] = x1; verts[i*8 + 5] = y1;
        verts[i*8 + 6] = x0; verts[i*8 + 7] = y1;
    }
    pthread_mutex_unlock(&vis_data->mutex);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);
}

static void render_bars(GLuint vao, VisData *vis_data, float width, float height) {
    glBindVertexArray(vao);
    for (int i = 0; i < NUM_BARS; ++i) {
        float bar_val;
        pthread_mutex_lock(&vis_data->mutex);
        bar_val = vis_data->magnitudes[i];
        pthread_mutex_unlock(&vis_data->mutex);

        float t = (float)i / (NUM_BARS-1);
        float r = 0.15f + 0.85f * t;
        float g = 1.0f - 0.7f * t;
        float b = 0.25f + 0.55f * (1.0f-t);
        glColor3f(r, g, b);
        glDrawArrays(GL_TRIANGLE_FAN, i*4, 4);
    }
}

void draw_audio_level(float level, int w, int h) {
    float bar_w = w * 0.15f;
    float bar_h = 20;
    float bar_x = 10;
    float bar_y = 10;
    float filled = level * bar_w;
    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0,w,0,h,-1,1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(0.18f,0.18f,0.18f);
    glBegin(GL_QUADS);
        glVertex2f(bar_x,        bar_y);
        glVertex2f(bar_x+bar_w,  bar_y);
        glVertex2f(bar_x+bar_w,  bar_y+bar_h);
        glVertex2f(bar_x,        bar_y+bar_h);
    glEnd();
    glColor3f(0.0f,0.9f,0.3f);
    glBegin(GL_QUADS);
        glVertex2f(bar_x,            bar_y);
        glVertex2f(bar_x+filled,     bar_y);
        glVertex2f(bar_x+filled,     bar_y+bar_h);
        glVertex2f(bar_x,            bar_y+bar_h);
    glEnd();
}

int main() {
    VisData vis_data;
    memset(&vis_data, 0, sizeof(vis_data));
    pthread_mutex_init(&vis_data.mutex, NULL);
    for (int i = 0; i < NUM_BARS; ++i) {
        vis_data.bin_maxes[i] = 1e-7f;
        vis_data.bin_ema[i] = 1.0f;
    }
    make_freq_table();

    AudioData ad;
    memset(&ad, 0, sizeof(ad));
    ad.vis_data = &vis_data;
    ad.inbuf_pos = 0;

    if (init_pipewire(&ad)) {
        fprintf(stderr, "Error: Failed to initialize PipeWire stream\n");
        return 1;
    }

    if (!glfwInit()) return 1;
    GLFWwindow *win = glfwCreateWindow(800, 400, "PipeWire FFT Visualizer", NULL, NULL);
    if (!win) return 1;
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    GLuint vaos[2], vbos[2];
    glGenVertexArrays(2, vaos);
    glGenBuffers(2, vbos);
    for (int i = 0; i < 2; ++i) {
        glBindVertexArray(vaos[i]);
        glBindBuffer(GL_ARRAY_BUFFER, vbos[i]);
        glBufferData(GL_ARRAY_BUFFER, NUM_BARS * NUM_VERTICES_PER_BAR * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    }
    int front = 0;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);

        interpolate_and_smooth(&vis_data);

        front = 1-front;
        update_vbo(vbos[front], &vis_data, (float)w, (float)h-32);

        glClearColor(0.04f,0.04f,0.08f,1); glClear(GL_COLOR_BUFFER_BIT);

        glViewport(0,32,w,h-32);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0,w,0,h-32,-1,1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        render_bars(vaos[front], &vis_data, (float)w, (float)h-32);

        float level;
        pthread_mutex_lock(&vis_data.mutex);
        level = vis_data.audio_level;
        pthread_mutex_unlock(&vis_data.mutex);

        draw_audio_level(level, w, h);

        glfwSwapBuffers(win);
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

