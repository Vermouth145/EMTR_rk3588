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

    const AVCodec *codec = avcodec_find_encoder_by_name(codec_name.c_str());
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
    // ffmpeg-rockchip 的 rkmpp 编码器支持 BGR24 输入，色彩转换由 MPP 硬件完成
    codec_ctx_->pix_fmt = in_pix_fmt_;
    codec_ctx_->bit_rate = 4 * 1000 * 1000;

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // MPP 编码器私有参数（码率控制等），需在 avcodec_open2 之前设置
    set_rkmpp_options();

    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        log_error("codec_open", ret);
        printf("[ERROR] ffmpeg: 若提示像素格式不支持，可能该 ffmpeg-rockchip 编码器未启用 BGR 输入，"
               "需改用 NV12 + RGA 预转换\n");
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

    frame_ = av_frame_alloc();
    if (!frame_) {
        printf("[ERROR] ffmpeg: failed to allocate frame\n");
        return false;
    }
    frame_->format = in_pix_fmt_;
    frame_->width = width_;
    frame_->height = height_;
    ret = av_frame_get_buffer(frame_, 32);
    if (ret < 0) {
        log_error("frame_get_buffer", ret);
        return false;
    }

    opened_ = true;
    return true;
}

void FfmpegEncoder::set_rkmpp_options()
{
    if (!codec_ctx_ || !codec_ctx_->priv_data) {
        return;
    }
    // 这些是 ffmpeg-rockchip rkmpp 编码器的私有选项；不同版本可能略有差异，
    // 设置失败仅告警、不致命（用 av_opt_set 的返回值判断）。
    struct { const char *key; const char *val; } opts[] = {
        {"rc_mode", "CBR"},        // 码率控制：CBR/VBR/CQP/AVBR
        {"profile", "high"},       // H.264 profile
        {"level", "40"},           // Level 4.0
    };
    for (const auto &o : opts) {
        int r = av_opt_set(codec_ctx_->priv_data, o.key, o.val, 0);
        if (r < 0) {
            printf("[WARN] ffmpeg: rkmpp option '%s=%s' 未生效(可能该版本不支持)\n", o.key, o.val);
        }
    }
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

bool FfmpegEncoder::fill_frame_from_bgr(const cv::Mat &bgr_frame)
{
    // 输入应为 8UC3 BGR，尺寸与编码器一致
    if (bgr_frame.type() != CV_8UC3 ||
        bgr_frame.cols != width_ || bgr_frame.rows != height_) {
        printf("[ERROR] ffmpeg: 输入帧格式/尺寸不匹配 (type=%d %dx%d, 期望 8UC3 %dx%d)\n",
               bgr_frame.type(), bgr_frame.cols, bgr_frame.rows, width_, height_);
        return false;
    }

    int ret = av_frame_make_writable(frame_);
    if (ret < 0) {
        log_error("frame_make_writable", ret);
        return false;
    }

    // BGR24 为单平面紧凑格式，逐行拷贝以兼容 OpenCV 与 AVFrame 各自的 stride
    av_image_copy_plane(frame_->data[0], frame_->linesize[0],
                        bgr_frame.data, static_cast<int>(bgr_frame.step[0]),
                        width_ * 3, height_);
    return true;
}

bool FfmpegEncoder::write(const cv::Mat &bgr_frame)
{
    if (!opened_ || !codec_ctx_ || !frame_) {
        return false;
    }
    if (bgr_frame.empty()) {
        return false;
    }

    if (!fill_frame_from_bgr(bgr_frame)) {
        return false;
    }

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
