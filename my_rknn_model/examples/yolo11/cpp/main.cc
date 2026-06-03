#include <stdio.h>
#include <sys/time.h>
#include <thread>
#include <queue>
#include <vector>
#include <string>
#include <cctype>
#include <unistd.h>
#include <algorithm>
#define _BASETSD_H

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "rknnPool.hpp"
#include "ThreadPool.hpp"
#include "ffmpeg_encoder.h"
#include "threat_analysis.h"

using std::queue;
using std::vector;

#define INFO_LOG(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)

static void print_usage()
{
    printf("使用方法:\n");
    printf("  ./rknn_yolo11_demo <模型路径> <视频路径>\n");
    printf("  ./rknn_yolo11_demo <模型路径> --input <视频/rtsp路径> [--output out.mp4] [--codec h264_rkmpp] [--fps 30]\n");
    printf("  ./rknn_yolo11_demo <模型路径> --camera <索引> [--output out.avi] [--codec XVID] [--fps 30]\n");
    printf("可选参数: --no-draw --no-track --use-rga --track-interval <N> --queue-size <N> --max-frames <N>\n");
    printf("          --json <path> --no-json\n");
}

static bool starts_with(const std::string &s, const std::string &prefix)
{
    return s.rfind(prefix, 0) == 0;
}

static bool parse_int(const std::string &s, int *out)
{
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    *out = std::atoi(s.c_str());
    return true;
}

static std::string to_lower(std::string s)
{
    for (char &c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

static int fourcc_from_codec(const std::string &codec, const std::string &output_path)
{
    if (codec.size() == 4) {
        return cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]);
    }
    std::string lower = to_lower(output_path);
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".avi") {
        return cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    }
    return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
}

struct AppOptions {
    std::string model_path;
    std::string input_path;
    std::string output_path = "out.mp4";
    std::string codec;
    bool use_camera = false;
    int camera_index = -1;
    bool draw_result = true;
    bool enable_track = true;
    bool enable_rga = false;
    int fps_override = 0;
    int max_frames = 0;
    int queue_size = 4;
    int track_interval = 1;
    bool enable_json = true;
    std::string json_path = "threat.jsonl";
};

int main(int argc, char **argv)
{
    AppOptions opts;

    if (argc < 3) {
        print_usage();
        return -1;
    }

    opts.model_path = argv[1];

    if (argc == 3) {
        opts.input_path = argv[2];
    } else {
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--input" || arg == "-i") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.input_path = argv[++i];
            } else if (arg == "--output" || arg == "-o") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.output_path = argv[++i];
            } else if (arg == "--codec") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.codec = argv[++i];
            } else if (arg == "--camera") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.use_camera = true;
                opts.camera_index = std::atoi(argv[++i]);
            } else if (arg == "--fps") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.fps_override = std::atoi(argv[++i]);
            } else if (arg == "--max-frames") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.max_frames = std::atoi(argv[++i]);
            } else if (arg == "--no-draw") {
                opts.draw_result = false;
            } else if (arg == "--no-track") {
                opts.enable_track = false;
            } else if (arg == "--use-rga") {
                opts.enable_rga = true;
            } else if (arg == "--queue-size") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.queue_size = std::atoi(argv[++i]);
                if (opts.queue_size <= 0) {
                    ERROR_LOG("Invalid queue size");
                    return -1;
                }
            } else if (arg == "--track-interval") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.track_interval = std::atoi(argv[++i]);
                if (opts.track_interval <= 0) {
                    ERROR_LOG("Invalid track interval");
                    return -1;
                }
            } else if (arg == "--no-json") {
                opts.enable_json = false;
            } else if (arg == "--json") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.enable_json = true;
                opts.json_path = argv[++i];
            } else if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            } else if (starts_with(arg, "camera:") || starts_with(arg, "cam:")) {
                std::string idx = arg.substr(arg.find(':') + 1);
                int cam = -1;
                if (!parse_int(idx, &cam)) { print_usage(); return -1; }
                opts.use_camera = true;
                opts.camera_index = cam;
            } else {
                // fallback for single input argument
                opts.input_path = arg;
            }
        }
    }

    if (!opts.use_camera && opts.input_path.empty()) {
        print_usage();
        return -1;
    }

    printf("模型: %s\n", opts.model_path.c_str());
    if (opts.use_camera) {
        printf("相机: %d\n", opts.camera_index);
    } else {
        printf("视频: %s\n", opts.input_path.c_str());
    }
    printf("输出: %s\n", opts.output_path.c_str());
    if (opts.enable_json) {
        printf("JSON日志: %s\n", opts.json_path.c_str());
    }

    cv::VideoCapture capture;
    if (opts.use_camera) {
        INFO_LOG("Opening camera...");
        capture.open(opts.camera_index);
    } else {
        bool is_rtsp = starts_with(opts.input_path, "rtsp://") ||
                       starts_with(opts.input_path, "rtsps://") ||
                       starts_with(opts.input_path, "http://") ||
                       starts_with(opts.input_path, "https://");
        INFO_LOG("Opening video...");
        capture.open(opts.input_path, is_rtsp ? cv::CAP_FFMPEG : cv::CAP_ANY);
    }

    if (!capture.isOpened()) {
        ERROR_LOG("Failed to open video source");
        return -1;
    }

    cv::Mat frame;
    if (!capture.read(frame) || frame.empty()) {
        ERROR_LOG("Failed to read first frame");
        return -1;
    }

    int video_width  = frame.cols;
    int video_height = frame.rows;
    double cap_fps   = capture.get(cv::CAP_PROP_FPS);
    int video_fps    = opts.fps_override > 0 ? opts.fps_override
                     : (cap_fps > 0 ? static_cast<int>(cap_fps) : 30);
    printf("Video info: %dx%d @ %dfps\n", video_width, video_height, video_fps);

    bool enable_writer = !(opts.output_path.empty() || opts.output_path == "-" || opts.output_path == "none");
    cv::VideoWriter writer;
    FfmpegEncoder ffmpeg_writer;
    bool use_ffmpeg = false;
    if (enable_writer) {
        std::string codec_lower = to_lower(opts.codec);
        if (!codec_lower.empty() && codec_lower == "h264_rkmpp") {
            use_ffmpeg = true;
            if (!ffmpeg_writer.open(opts.output_path, video_width, video_height, video_fps, "h264_rkmpp")) {
                ERROR_LOG("Failed to open FFmpeg encoder");
                return -1;
            }
        } else {
            int fourcc = fourcc_from_codec(opts.codec, opts.output_path);
            writer.open(opts.output_path, fourcc, video_fps, cv::Size(video_width, video_height));
            if (!writer.isOpened()) {
                ERROR_LOG("Failed to open video writer");
                return -1;
            }
        }
    }

    ThreatAnalyzer threat_analyzer(static_cast<double>(video_fps));
    ThreatLogger threat_logger;
    if (opts.enable_json) {
        if (!threat_logger.open(opts.json_path)) {
            ERROR_LOG("Failed to open JSON log: %s", opts.json_path.c_str());
            return -1;
        }
    }

    INFO_LOG("Creating RKNN instance...");
    rknn_lite *model = new rknn_lite(const_cast<char *>(opts.model_path.c_str()), 0, 0);
    model->set_track_interval(opts.track_interval);

    std::deque<cv::Mat> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_not_empty;
    std::condition_variable queue_not_full;
    std::atomic<bool> capture_done(false);
    std::atomic<bool> stop_requested(false);

    // push first frame into queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        frame_queue.push_back(frame.clone());
    }
    queue_not_empty.notify_one();

    std::thread capture_thread([&]() {
        cv::Mat cap_frame;
        while (!stop_requested.load()) {
            if (!capture.read(cap_frame) || cap_frame.empty()) {
                break;
            }
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_not_full.wait(lock, [&]() {
                return stop_requested.load() || frame_queue.size() < static_cast<size_t>(opts.queue_size);
            });
            if (stop_requested.load()) {
                break;
            }
            frame_queue.push_back(cap_frame.clone());
            lock.unlock();
            queue_not_empty.notify_one();
        }
        capture_done.store(true);
        queue_not_empty.notify_all();
    });

    struct timeval time_now;
    gettimeofday(&time_now, nullptr);
    auto initTime = time_now.tv_sec * 1000 + time_now.tv_usec / 1000;
    long tmpTime, lopTime = initTime;
    int frames = 0;

    INFO_LOG("Starting processing...");
    while (true)
    {
        cv::Mat cur;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_not_empty.wait(lock, [&]() {
                return capture_done.load() || !frame_queue.empty();
            });
            if (frame_queue.empty() && capture_done.load()) {
                break;
            }
            if (!frame_queue.empty()) {
                cur = frame_queue.front();
                frame_queue.pop_front();
                queue_not_full.notify_one();
            }
        }

        if (cur.empty()) {
            if (capture_done.load()) {
                break;
            }
            continue;
        }

        model->ori_img = cur;

        if (model->interf(opts.draw_result, opts.enable_track, opts.enable_rga) != 0) {
            ERROR_LOG("Inference failed at frame %d", frames);
            break;
        }

        if (opts.enable_track) {
            int frame_index = model->get_frame_index();
            const auto &tracks = model->get_last_tracks();
            for (const auto &t : tracks) {
                ThreatBBox box;
                box.x = static_cast<int>(t.tlwh[0]);
                box.y = static_cast<int>(t.tlwh[1]);
                box.w = static_cast<int>(t.tlwh[2]);
                box.h = static_cast<int>(t.tlwh[3]);

                ThreatResult result = threat_analyzer.update(frame_index, t.track_id, t.label, box);

                char threat_text[160];
                snprintf(threat_text, sizeof(threat_text), "%sD%.1fm V%.1fkm/h T%.2f %s",
                         result.is_dangerous ? "[DANGER] " : "",
                         result.distance_m, result.speed_kmh, result.threat_score, result.type);

                // 危险目标红色，否则黄色（对齐 video_alarm.py 的红/绿配色思路）
                cv::Scalar threat_color = result.is_dangerous ? cv::Scalar(0, 0, 255)
                                                              : cv::Scalar(0, 255, 255);
                int text_y = std::max(16, box.y + box.h + 16);
                cv::putText(model->ori_img, threat_text,
                            cv::Point(box.x, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            threat_color, 1);

                if (threat_logger.enabled()) {
                    ThreatLogRecord rec;
                    rec.frame_index = frame_index;
                    rec.track_id = t.track_id;
                    rec.label = t.label;
                    rec.x = box.x;
                    rec.y = box.y;
                    rec.w = box.w;
                    rec.h = box.h;
                    rec.distance_m = result.distance_m;
                    rec.speed_kmh = result.speed_kmh;
                    rec.threat_score = result.threat_score;
                    rec.is_dangerous = result.is_dangerous;
                    rec.type = result.type;
                    threat_logger.log(rec);
                }
            }
            threat_analyzer.purge_stale(frame_index, video_fps * 2);
        }

        if (enable_writer) {
            if (use_ffmpeg) {
                if (!ffmpeg_writer.write(model->ori_img)) {
                    ERROR_LOG("FFmpeg encode failed at frame %d", frames);
                    break;
                }
            } else {
                writer.write(model->ori_img);
            }
        }
        frames++;

        if (opts.max_frames > 0 && frames >= opts.max_frames) {
            stop_requested.store(true);
            queue_not_full.notify_all();
            break;
        }

        if (frames % 60 == 0) {
            gettimeofday(&time_now, nullptr);
            tmpTime = time_now.tv_sec * 1000 + time_now.tv_usec / 1000;
            printf("[FPS] 60帧平均: %.2f FPS\n", 60000.0 / (float)(tmpTime - lopTime));
            lopTime = tmpTime;
        }
    }

    stop_requested.store(true);
    queue_not_full.notify_all();
    if (capture_thread.joinable()) {
        capture_thread.join();
    }

    gettimeofday(&time_now, nullptr);
    long totalTime = time_now.tv_sec * 1000 + time_now.tv_usec / 1000 - initTime;
    float total_fps = (totalTime > 0) ? (float)frames / (float)totalTime * 1000.0f : 0;
    printf("\n[SUMMARY] 总帧数: %d, 总耗时: %.2fs, 平均帧率: %.2f FPS\n",
           frames, totalTime / 1000.0f, total_fps);

    delete model;
    capture.release();
    if (enable_writer) {
        if (use_ffmpeg) {
            ffmpeg_writer.close();
        } else {
            writer.release();
        }
        printf("[SUMMARY] 视频保存完成: %s\n", opts.output_path.c_str());
    }
    if (threat_logger.enabled()) {
        threat_logger.close();
    }
    return 0;
}
