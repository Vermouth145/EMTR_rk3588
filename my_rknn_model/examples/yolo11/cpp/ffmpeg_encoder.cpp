#include "ffmpeg_encoder.h"

#include <stdio.h>

FfmpegEncoder::FfmpegEncoder() = default;

FfmpegEncoder::~FfmpegEncoder()
{
    close();
}

bool FfmpegEncoder::open(const std::string &output_path, int width, int height, int fps, const std::string &codec_name)
{
    if (opened_) {
        return false;
    }

    width_ = width;
    height_ = height;
    fps_ = fps > 0 ? fps : 30;

    AVCodec *codec = avcodec_find_encoder_by_name(codec_name.c_str());
    if (!codec) {
        printf("[ERROR] ffmpeg: encoder not found: %s\n", codec_name.c_str());
        return false;
    }

    int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, nullptr, output_path.c_str());
    if (ret < 0 || !fmt_ctx_) {
        log_error("alloc_output_context", ret);
        return false;
    }

    stream_ = avformat_new_stream(fmt_ctx_, codec);
    if (!stream_) {
        printf("[ERROR] ffmpeg: failed to create stream\n");
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        printf("[ERROR] ffmpeg: failed to allocate codec context\n");
        return false;
    }

    codec_ctx_->codec_id = codec->id;
    codec_ctx_->width = width_;
    codec_ctx_->height = height_;
    codec_ctx_->time_base = AVRational{1, fps_};
    codec_ctx_->framerate = AVRational{fps_, 1};
    codec_ctx_->gop_size = fps_;
    codec_ctx_->max_b_frames = 0;
    codec_ctx_->pix_fmt = AV_PIX_FMT_NV12;
    codec_ctx_->bit_rate = 4 * 1000 * 1000;

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        log_error("codec_open", ret);
        return false;
    }

    ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
    if (ret < 0) {
        log_error("parameters_from_context", ret);
        return false;
    }

    stream_->time_base = codec_ctx_->time_base;

    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&fmt_ctx_->pb, output_path.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            log_error("avio_open", ret);
            return false;
        }
    }

    ret = avformat_write_header(fmt_ctx_, nullptr);
    if (ret < 0) {
        log_error("write_header", ret);
        return false;
    }

    if (!init_sws(width_, height_)) {
        return false;
    }

    opened_ = true;
    return true;
}

bool FfmpegEncoder::init_sws(int width, int height)
{
    sws_ctx_ = sws_getContext(width, height, AV_PIX_FMT_BGR24,
                              width, height, AV_PIX_FMT_NV12,
                              SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        printf("[ERROR] ffmpeg: failed to create sws context\n");
        return false;
    }

    frame_ = av_frame_alloc();
    if (!frame_) {
        printf("[ERROR] ffmpeg: failed to allocate frame\n");
        return false;
    }

    frame_->format = codec_ctx_->pix_fmt;
    frame_->width = width;
    frame_->height = height;

    int ret = av_frame_get_buffer(frame_, 32);
    if (ret < 0) {
        log_error("frame_get_buffer", ret);
        return false;
    }
    return true;
}

bool FfmpegEncoder::encode_frame(AVFrame *frame)
{
    int ret = avcodec_send_frame(codec_ctx_, frame);
    if (ret < 0) {
        log_error("send_frame", ret);
        return false;
    }

    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "[ERROR] ffmpeg: av_packet_alloc failed\n");
        return false;
    }
    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx_, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) {
            log_error("receive_packet", ret);
            av_packet_free(&pkt);
            return false;
        }

        av_packet_rescale_ts(pkt, codec_ctx_->time_base, stream_->time_base);
        pkt->stream_index = stream_->index;

        ret = av_interleaved_write_frame(fmt_ctx_, pkt);
        av_packet_unref(pkt);
        if (ret < 0) {
            log_error("write_frame", ret);
            av_packet_free(&pkt);
            return false;
        }
    }
    av_packet_free(&pkt);

    return true;
}

bool FfmpegEncoder::write(const cv::Mat &bgr_frame)
{
    if (!opened_ || !codec_ctx_ || !sws_ctx_ || !frame_) {
        return false;
    }

    if (bgr_frame.empty()) {
        return false;
    }

    cv::Mat input = bgr_frame;
    if (!input.isContinuous()) {
        input = input.clone();
    }

    const uint8_t *in_data[1] = { input.data };
    int in_linesize[1] = { static_cast<int>(input.step[0]) };

    int ret = av_frame_make_writable(frame_);
    if (ret < 0) {
        log_error("frame_make_writable", ret);
        return false;
    }

    sws_scale(sws_ctx_, in_data, in_linesize, 0, height_, frame_->data, frame_->linesize);

    frame_->pts = frame_index_++;

    return encode_frame(frame_);
}

void FfmpegEncoder::close()
{
    if (!opened_) {
        return;
    }

    encode_frame(nullptr);

    if (fmt_ctx_) {
        av_write_trailer(fmt_ctx_);
    }

    if (frame_) {
        av_frame_free(&frame_);
        frame_ = nullptr;
    }

    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }

    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }

    if (fmt_ctx_) {
        if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&fmt_ctx_->pb);
        }
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
    }

    opened_ = false;
}

void FfmpegEncoder::log_error(const char *stage, int err) const
{
    char errbuf[128];
    av_strerror(err, errbuf, sizeof(errbuf));
    printf("[ERROR] ffmpeg: %s failed: %s\n", stage, errbuf);
}

