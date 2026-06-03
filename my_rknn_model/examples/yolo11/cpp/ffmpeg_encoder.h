#pragma once

#include <string>

#include "opencv2/core.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}

class FfmpegEncoder {
public:
    FfmpegEncoder();
    ~FfmpegEncoder();

    bool open(const std::string &output_path, int width, int height, int fps, const std::string &codec_name);
    bool write(const cv::Mat &bgr_frame);
    void close();

private:
    bool init_sws(int width, int height);
    bool encode_frame(AVFrame *frame);
    void log_error(const char *stage, int err) const;

    AVFormatContext *fmt_ctx_ = nullptr;
    AVCodecContext *codec_ctx_ = nullptr;
    AVStream *stream_ = nullptr;
    SwsContext *sws_ctx_ = nullptr;
    AVFrame *frame_ = nullptr;
    int frame_index_ = 0;
    int width_ = 0;
    int height_ = 0;
    int fps_ = 30;
    bool opened_ = false;
};

