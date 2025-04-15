#include <iostream>
#include <random>
#include <vector>
#include <string>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswresample/swresample.h>
}

static void log_error(const char* msg, int err) {
    char buf[256];
    av_strerror(err, buf, sizeof(buf));
    std::cerr << msg << ": " << buf << std::endl;
    exit(1);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " input.mp3 output.mp4\n";
        return 1;
    }
    const char* in_filename = argv[1];
    const char* out_filename = argv[2];
    avformat_network_init();

    AVFormatContext* in_fmt = nullptr;
    if (avformat_open_input(&in_fmt, in_filename, nullptr, nullptr) < 0)
        log_error("Could not open input", -1);
    if (avformat_find_stream_info(in_fmt, nullptr) < 0)
        log_error("Could not find stream info", -1);

    int audio_stream_index = -1;
    for (unsigned i = 0; i < in_fmt->nb_streams; ++i) {
        if (in_fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_index = i;
            break;
        }
    }
    if (audio_stream_index < 0)
        log_error("No audio stream found", -1);

    AVCodecParameters* audio_codecpar = in_fmt->streams[audio_stream_index]->codecpar;
    const AVCodec* audio_codec = avcodec_find_decoder(audio_codecpar->codec_id);
    AVCodecContext* audio_codec_ctx = avcodec_alloc_context3(audio_codec);
    avcodec_parameters_to_context(audio_codec_ctx, audio_codecpar);
    if (avcodec_open2(audio_codec_ctx, audio_codec, nullptr) < 0)
        log_error("Could not open audio codec", -1);

    SwrContext* swr = swr_alloc();
    av_opt_set_chlayout(swr, "in_chlayout", &audio_codec_ctx->ch_layout, 0);
    AVChannelLayout mono_layout = AV_CHANNEL_LAYOUT_MONO;
    av_opt_set_chlayout(swr, "out_chlayout", &mono_layout, 0);
    av_opt_set_int(swr, "in_sample_rate", audio_codec_ctx->sample_rate, 0);
    av_opt_set_int(swr, "out_sample_rate", 44100, 0);
    av_opt_set_sample_fmt(swr, "in_sample_fmt", audio_codec_ctx->sample_fmt, 0);
    av_opt_set_sample_fmt(swr, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);
    swr_init(swr);

    AVFormatContext* out_fmt = nullptr;
    avformat_alloc_output_context2(&out_fmt, nullptr, nullptr, out_filename);

    const AVCodec* video_codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
    AVStream* video_stream = avformat_new_stream(out_fmt, video_codec);

    AVCodecContext* video_codec_ctx = avcodec_alloc_context3(video_codec);
    video_codec_ctx->codec_id = AV_CODEC_ID_MPEG4;
    video_codec_ctx->bit_rate = 400000;
    video_codec_ctx->width = 320;
    video_codec_ctx->height = 240;
    video_codec_ctx->time_base = {1, 30};
    video_stream->time_base = video_codec_ctx->time_base;
    video_codec_ctx->framerate = {30, 1};
    video_codec_ctx->gop_size = 12;
    video_codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

    if (out_fmt->oformat->flags & AVFMT_GLOBALHEADER)
        video_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(video_codec_ctx, video_codec, nullptr) < 0)
        log_error("Could not open video codec", -1);

    avcodec_parameters_from_context(video_stream->codecpar, video_codec_ctx);

    const AVCodec* audio_enc_codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    AVStream* audio_stream = avformat_new_stream(out_fmt, audio_enc_codec);
    AVCodecContext* audio_enc_ctx = avcodec_alloc_context3(audio_enc_codec);
    audio_enc_ctx->codec_id = AV_CODEC_ID_AAC;
    audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP;
    audio_enc_ctx->bit_rate = 128000;
    audio_enc_ctx->sample_rate = 44100;
    av_channel_layout_copy(&audio_enc_ctx->ch_layout, &mono_layout);
    audio_stream->time_base = {1, audio_enc_ctx->sample_rate};
    if (out_fmt->oformat->flags & AVFMT_GLOBALHEADER)
        audio_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (avcodec_open2(audio_enc_ctx, audio_enc_codec, nullptr) < 0)
        log_error("Could not open output audio encoder", -1);
    avcodec_parameters_from_context(audio_stream->codecpar, audio_enc_ctx);

    if (!(out_fmt->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&out_fmt->pb, out_filename, AVIO_FLAG_WRITE) < 0)
            log_error("Could not open output file", -1);
    }

    if (avformat_write_header(out_fmt, nullptr) < 0)
        log_error("Could not write header", -1);

    AVPacket* pkt = av_packet_alloc();
    AVFrame* audio_frame = av_frame_alloc();
    audio_frame->sample_rate = 44100;
    audio_frame->format = AV_SAMPLE_FMT_S16;
    av_channel_layout_copy(&audio_frame->ch_layout, &mono_layout);
    audio_frame->nb_samples = 1024;
    av_frame_get_buffer(audio_frame, 0);

    AVFrame* video_frame = av_frame_alloc();
    video_frame->format = video_codec_ctx->pix_fmt;
    video_frame->width = video_codec_ctx->width;
    video_frame->height = video_codec_ctx->height;
    av_frame_get_buffer(video_frame, 0);

    int64_t video_pts = 0, audio_pts = 0;
    int finished = 0;
    int samples_written = 0;

    std::default_random_engine rnd;
    std::uniform_int_distribution<> dist(0, 255);

    std::vector<int16_t> audio_buf;
    while (av_read_frame(in_fmt, pkt) >= 0) {
        if (pkt->stream_index == audio_stream_index) {
            if (avcodec_send_packet(audio_codec_ctx, pkt) == 0) {
                while (avcodec_receive_frame(audio_codec_ctx, audio_frame) == 0) {
                    uint8_t* out_samples;
                    int out_linesize;
                    int out_count = av_rescale_rnd(swr_get_delay(swr, audio_codec_ctx->sample_rate) + audio_frame->nb_samples,
                                                   44100, audio_codec_ctx->sample_rate, AV_ROUND_UP);
                    av_samples_alloc(&out_samples, &out_linesize, 1, out_count, AV_SAMPLE_FMT_S16, 0);
                    int conv = swr_convert(swr, &out_samples, out_count, (const uint8_t**)audio_frame->extended_data, audio_frame->nb_samples);
                    int16_t* data = (int16_t*)out_samples;
                    for (int i = 0; i < conv; ++i)
                        audio_buf.push_back(data[i]);
                    av_freep(&out_samples);
                    while ((int)audio_buf.size() >= 1024) {
                        AVFrame* enc_frame = av_frame_alloc();
                        enc_frame->nb_samples = 1024;
                        enc_frame->format = audio_enc_ctx->sample_fmt;
                        av_channel_layout_copy(&enc_frame->ch_layout, &audio_enc_ctx->ch_layout);
                        av_frame_get_buffer(enc_frame, 0);
                        for (int i = 0; i < 1024; ++i)
                            ((float*)enc_frame->extended_data[0])[i] = audio_buf[i] / 32768.f;
                        audio_buf.erase(audio_buf.begin(), audio_buf.begin() + 1024);
                        enc_frame->pts = audio_pts;
                        audio_pts += enc_frame->nb_samples;
                        if (avcodec_send_frame(audio_enc_ctx, enc_frame) == 0) {
                            AVPacket* outpkt = av_packet_alloc();
                            while (avcodec_receive_packet(audio_enc_ctx, outpkt) == 0) {
                                av_packet_rescale_ts(outpkt, audio_enc_ctx->time_base, audio_stream->time_base);
                                outpkt->stream_index = audio_stream->index;
                                av_interleaved_write_frame(out_fmt, outpkt);
                                av_packet_unref(outpkt);
                            }
                            av_packet_free(&outpkt);
                        }
                        av_frame_free(&enc_frame);

                        video_frame->pts = video_pts++;
                        av_frame_make_writable(video_frame);
                        int c = dist(rnd), cr = dist(rnd), cb = dist(rnd);
                        for (int j = 0; j < video_codec_ctx->height; ++j)
                            memset(video_frame->data[0] + j*video_frame->linesize[0], c, video_codec_ctx->width);
                        for (int j = 0; j < video_codec_ctx->height/2; ++j) {
                            memset(video_frame->data[1] + j*video_frame->linesize[1], cr, video_codec_ctx->width/2);
                            memset(video_frame->data[2] + j*video_frame->linesize[2], cb, video_codec_ctx->width/2);
                        }
                        if (avcodec_send_frame(video_codec_ctx, video_frame) == 0) {
                            AVPacket* voutpkt = av_packet_alloc();
                            while (avcodec_receive_packet(video_codec_ctx, voutpkt) == 0) {
                                av_packet_rescale_ts(voutpkt, video_codec_ctx->time_base, video_stream->time_base);
                                voutpkt->stream_index = video_stream->index;
                                av_interleaved_write_frame(out_fmt, voutpkt);
                                av_packet_unref(voutpkt);
                            }
                            av_packet_free(&voutpkt);
                        }
                    }
                }
            }
        }
        av_packet_unref(pkt);
    }
    avcodec_send_packet(audio_codec_ctx, nullptr);
    while (avcodec_receive_frame(audio_codec_ctx, audio_frame) == 0) {
        uint8_t* out_samples;
        int out_linesize;
        int out_count = av_rescale_rnd(swr_get_delay(swr, audio_codec_ctx->sample_rate) + audio_frame->nb_samples,
                                       44100, audio_codec_ctx->sample_rate, AV_ROUND_UP);
        av_samples_alloc(&out_samples, &out_linesize, 1, out_count, AV_SAMPLE_FMT_S16, 0);
        int conv = swr_convert(swr, &out_samples, out_count, (const uint8_t**)audio_frame->extended_data, audio_frame->nb_samples);
        int16_t* data = (int16_t*)out_samples;
        for (int i = 0; i < conv; ++i)
            audio_buf.push_back(data[i]);
        av_freep(&out_samples);
    }
    while ((int)audio_buf.size() >= 1024) {
        AVFrame* enc_frame = av_frame_alloc();
        enc_frame->nb_samples = 1024;
        enc_frame->format = audio_enc_ctx->sample_fmt;
        av_channel_layout_copy(&enc_frame->ch_layout, &audio_enc_ctx->ch_layout);
        av_frame_get_buffer(enc_frame, 0);
        for (int i = 0; i < 1024; ++i)
            ((float*)enc_frame->extended_data[0])[i] = audio_buf[i] / 32768.f;
        audio_buf.erase(audio_buf.begin(), audio_buf.begin() + 1024);
        enc_frame->pts = audio_pts;
        audio_pts += enc_frame->nb_samples;
        if (avcodec_send_frame(audio_enc_ctx, enc_frame) == 0) {
            AVPacket* outpkt = av_packet_alloc();
            while (avcodec_receive_packet(audio_enc_ctx, outpkt) == 0) {
                av_packet_rescale_ts(outpkt, audio_enc_ctx->time_base, audio_stream->time_base);
                outpkt->stream_index = audio_stream->index;
                av_interleaved_write_frame(out_fmt, outpkt);
                av_packet_unref(outpkt);
            }
            av_packet_free(&outpkt);
        }
        av_frame_free(&enc_frame);

        video_frame->pts = video_pts++;
        av_frame_make_writable(video_frame);
        int c = dist(rnd), cr = dist(rnd), cb = dist(rnd);
        for (int j = 0; j < video_codec_ctx->height; ++j)
            memset(video_frame->data[0] + j*video_frame->linesize[0], c, video_codec_ctx->width);
        for (int j = 0; j < video_codec_ctx->height/2; ++j) {
            memset(video_frame->data[1] + j*video_frame->linesize[1], cr, video_codec_ctx->width/2);
            memset(video_frame->data[2] + j*video_frame->linesize[2], cb, video_codec_ctx->width/2);
        }
        if (avcodec_send_frame(video_codec_ctx, video_frame) == 0) {
            AVPacket* voutpkt = av_packet_alloc();
            while (avcodec_receive_packet(video_codec_ctx, voutpkt) == 0) {
                av_packet_rescale_ts(voutpkt, video_codec_ctx->time_base, video_stream->time_base);
                voutpkt->stream_index = video_stream->index;
                av_interleaved_write_frame(out_fmt, voutpkt);
                av_packet_unref(voutpkt);
            }
            av_packet_free(&voutpkt);
        }
    }
    av_write_trailer(out_fmt);

    av_frame_free(&audio_frame);
    av_frame_free(&video_frame);
    avcodec_free_context(&audio_codec_ctx);
    avcodec_free_context(&audio_enc_ctx);
    avcodec_free_context(&video_codec_ctx);
    swr_free(&swr);
    avformat_close_input(&in_fmt);
    avio_closep(&out_fmt->pb);
    avformat_free_context(out_fmt);
    av_packet_free(&pkt);
    return 0;
}
