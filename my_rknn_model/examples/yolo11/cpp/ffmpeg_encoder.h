#pragma once

#include <string>

#include "opencv2/core.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/error.h>
#include <libavutil/pixfmt.h>
}

// 基于 ffmpeg-rockchip 的 MPP 硬件视频编码封装。
// 关键点：ffmpeg-rockchip 的 *_rkmpp 编码器可直接接收 BGR24 等 RGB 输入，
// 由 MPP/RGA 在硬件内部完成色彩空间转换，因此无需 swscale 在 CPU 上做 BGR->NV12。
class FfmpegEncoder {
public:
    FfmpegEncoder();
    ~FfmpegEncoder();

    // codec_name 例：h264_rkmpp / hevc_rkmpp
    bool open(const std::string &output_path, int width, int height, int fps, const std::string &codec_name);
    bool write(const cv::Mat &bgr_frame);
    void close();

private:
    bool fill_frame_from_bgr(const cv::Mat &bgr_frame);
    bool encode_frame(AVFrame *frame);
    void log_error(const char *stage, int err) const;
    void set_rkmpp_options();

    AVFormatContext *fmt_ctx_ = nullptr;
    AVCodecContext *codec_ctx_ = nullptr;
    AVStream *stream_ = nullptr;
    AVFrame *frame_ = nullptr;
    int frame_index_ = 0;
    int width_ = 0;
    int height_ = 0;
    int fps_ = 30;
    // 编码器输入像素格式：BGR24 直喂硬件编码器（MPP 内部做 CSC）
    AVPixelFormat in_pix_fmt_ = AV_PIX_FMT_BGR24;
    bool opened_ = false;
};
