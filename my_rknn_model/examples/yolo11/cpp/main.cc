#include <stdio.h>
#include <sys/time.h>
#include <thread>
#include <deque>
#include <queue>
#include <vector>
#include <string>
#include <cctype>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <utility>
#define _BASETSD_H

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "rknnPool.hpp"
#include "ThreadPool.hpp"
#include "ffmpeg_encoder.h"
#include "threat_analysis.h"

using std::vector;

#define INFO_LOG(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)

static void print_usage()
{
    printf("使用方法:\n");
    printf("  ./rknn_yolo11_demo <模型路径> <视频路径>\n");
    printf("  ./rknn_yolo11_demo <模型路径> --input <视频/rtsp路径> [--output out.mp4] [--codec h264_rkmpp] [--fps 30]\n");
    printf("  ./rknn_yolo11_demo <模型路径> --camera <索引> [--output out.avi] [--codec XVID] [--fps 30]\n");
    printf("可选参数: --no-draw --no-track --track-interval <N> --queue-size <N> --max-frames <N>\n");
    printf("          --json <path> --no-json\n");
    printf("          --npu-cores <1-3>  (NPU 推理核数，默认 3)\n");
    printf("          --no-rga           (关闭 RGA 硬件预处理，默认开启)\n");
    printf("          --encode-queue <N> (编码线程队列长度，默认 8)\n");
    printf("          --drop-frames | --no-drop  (实时源队满丢帧；默认相机/RTSP丢帧、文件不丢)\n");
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
    bool enable_rga = true;          // 默认开启 RGA 硬件预处理
    int fps_override = 0;
    int max_frames = 0;
    int queue_size = 4;
    int track_interval = 1;
    bool enable_json = true;
    std::string json_path = "threat.jsonl";
    int npu_cores = 3;               // NPU 推理核数（RK3588 共 3 核）
    int encode_queue = 8;            // 编码线程队列长度
    int drop_mode = -1;              // -1=自动(实时源丢/文件不丢) 0=不丢 1=丢
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
            } else if (arg == "--no-rga") {
                opts.enable_rga = false;
            } else if (arg == "--npu-cores") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.npu_cores = std::atoi(argv[++i]);
                if (opts.npu_cores < 1 || opts.npu_cores > 3) {
                    ERROR_LOG("Invalid npu-cores (1-3)");
                    return -1;
                }
            } else if (arg == "--encode-queue") {
                if (i + 1 >= argc) { print_usage(); return -1; }
                opts.encode_queue = std::atoi(argv[++i]);
                if (opts.encode_queue <= 0) {
                    ERROR_LOG("Invalid encode-queue");
                    return -1;
                }
            } else if (arg == "--drop-frames") {
                opts.drop_mode = 1;
            } else if (arg == "--no-drop") {
                opts.drop_mode = 0;
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
    bool is_rtsp = false;
    if (opts.use_camera) {
        INFO_LOG("Opening camera...");
        capture.open(opts.camera_index);
    } else {
        is_rtsp = starts_with(opts.input_path, "rtsp://") ||
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

    // 是否实时源（相机/RTSP）：决定默认丢帧策略
    const bool realtime_source = opts.use_camera || is_rtsp;
    const bool drop_frames = (opts.drop_mode == 1) ||
                             (opts.drop_mode == -1 && realtime_source);
    printf("配置: NPU核数=%d RGA=%s 丢帧=%s 编码队列=%d\n",
           opts.npu_cores, opts.enable_rga ? "on" : "off",
           drop_frames ? "on" : "off", opts.encode_queue);

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

    INFO_LOG("Creating RKNN pool (%d cores)...", opts.npu_cores);
    RknnPool pool(opts.model_path.c_str(), opts.npu_cores, opts.enable_rga);
    pool.warmup();   // 预热标签加载，避免首批并发任务竞争 lazy init
    const int pool_size = static_cast<int>(pool.size());

    // 跟踪/威胁/绘制全部在主消费线程，单实例、按帧序，正确性等价于原串行版
    BYTETracker tracker(30, 90);
    std::vector<STrack> last_tracks;

    // ---- 采集线程：解码 → 有界帧队列（可选丢最旧帧） ----
    std::deque<cv::Mat> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_not_empty;
    std::condition_variable queue_not_full;
    std::atomic<bool> capture_done(false);
    std::atomic<bool> stop_requested(false);

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
            if (drop_frames) {
                // 实时源：队满则丢最旧帧，避免延迟累积
                while (frame_queue.size() >= static_cast<size_t>(opts.queue_size)) {
                    frame_queue.pop_front();
                }
            } else {
                // 文件源：队满则阻塞，保证不丢帧
                queue_not_full.wait(lock, [&]() {
                    return stop_requested.load() ||
                           frame_queue.size() < static_cast<size_t>(opts.queue_size);
                });
                if (stop_requested.load()) break;
            }
            frame_queue.push_back(cap_frame.clone());
            lock.unlock();
            queue_not_empty.notify_one();
        }
        capture_done.store(true);
        queue_not_empty.notify_all();
    });

    // 阻塞取一帧；返回 false 表示采集结束且队列已空
    auto get_frame = [&](cv::Mat &out) -> bool {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_not_empty.wait(lock, [&]() {
            return capture_done.load() || !frame_queue.empty();
        });
        if (frame_queue.empty()) {
            return false;   // capture_done 且空
        }
        out = frame_queue.front();
        frame_queue.pop_front();
        queue_not_full.notify_one();
        return true;
    };

    // ---- 编码线程：消费画好的帧 → 写文件（与推理重叠） ----
    std::deque<cv::Mat> encode_queue;
    std::mutex enc_mutex;
    std::condition_variable enc_not_empty;
    std::condition_variable enc_not_full;
    std::atomic<bool> encode_done_input(false);
    std::atomic<bool> encode_failed(false);
    std::thread encode_thread;
    if (enable_writer) {
        encode_thread = std::thread([&]() {
            while (true) {
                cv::Mat img;
                {
                    std::unique_lock<std::mutex> lock(enc_mutex);
                    enc_not_empty.wait(lock, [&]() {
                        return encode_done_input.load() || !encode_queue.empty();
                    });
                    if (encode_queue.empty()) {
                        if (encode_done_input.load()) break;
                        continue;
                    }
                    img = encode_queue.front();
                    encode_queue.pop_front();
                    enc_not_full.notify_one();
                }
                if (use_ffmpeg) {
                    if (!ffmpeg_writer.write(img)) {
                        ERROR_LOG("FFmpeg encode failed");
                        encode_failed.store(true);
                        break;
                    }
                } else {
                    writer.write(img);
                }
            }
        });
    }

    auto push_encode = [&](const cv::Mat &img) {
        if (!enable_writer) return;
        std::unique_lock<std::mutex> lock(enc_mutex);
        enc_not_full.wait(lock, [&]() {
            return encode_failed.load() ||
                   encode_queue.size() < static_cast<size_t>(opts.encode_queue);
        });
        if (encode_failed.load()) return;
        encode_queue.push_back(img);
        lock.unlock();
        enc_not_empty.notify_one();
    };

    struct timeval time_now;
    gettimeofday(&time_now, nullptr);
    auto initTime = time_now.tv_sec * 1000 + time_now.tv_usec / 1000;
    long tmpTime, lopTime = initTime;
    int frames = 0;

    // STATS 聚合（来自各帧 InferTiming）
    double acc_pre = 0.0, acc_infer = 0.0, acc_post = 0.0;
    int stat_frames = 0;

    std::queue<std::future<InferResult>> futs;
    int submit_index = 0;
    bool input_drained = false;

    INFO_LOG("Starting processing...");
    while (true)
    {
        // 1) 填充推理流水线：保持最多 pool_size 个在途任务
        while (!input_drained && static_cast<int>(futs.size()) < pool_size) {
            cv::Mat f;
            if (!get_frame(f)) {
                input_drained = true;
                break;
            }
            futs.push(pool.submit(submit_index++, std::move(f)));
        }

        if (futs.empty()) {
            break;  // 无在途任务且输入耗尽
        }

        // 2) 按帧序取最旧结果
        InferResult r = futs.front().get();
        futs.pop();
        if (!r.ok) {
            ERROR_LOG("Inference failed at frame %d", r.frame_index);
            break;
        }

        // 3) 跟踪（单实例、顺序）+ 威胁分析 + 绘制
        if (opts.enable_track) {
            bool do_track = (opts.track_interval <= 1) ||
                            (r.frame_index % opts.track_interval == 0);
            if (do_track) {
                last_tracks = tracker.update(r.objs);
            }

            for (const auto &t : last_tracks) {
                int x = int(t.tlwh[0]);
                int y = int(t.tlwh[1]);
                int w = int(t.tlwh[2]);
                int h = int(t.tlwh[3]);

                ThreatBBox box;
                box.x = x; box.y = y; box.w = w; box.h = h;
                ThreatResult result = threat_analyzer.update(r.frame_index, t.track_id, t.label, box);

                if (opts.draw_result) {
                    cv::Scalar color = tracker.get_color(t.track_id);
                    const char *label_name = get_label_name(t.label);
                    char id_text[128];
                    snprintf(id_text, sizeof(id_text), "%s ID:%d %.1f%%",
                             label_name, t.track_id, t.score * 100.0f);
                    cv::rectangle(r.img, cv::Point(x, y), cv::Point(x + w, y + h), color, 2);
                    cv::putText(r.img, id_text, cv::Point(x, y - 6),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, color);

                    char threat_text[160];
                    snprintf(threat_text, sizeof(threat_text), "%sD%.1fm V%.1fkm/h T%.2f %s",
                             result.is_dangerous ? "[DANGER] " : "",
                             result.distance_m, result.speed_kmh, result.threat_score, result.type);
                    cv::Scalar threat_color = result.is_dangerous ? cv::Scalar(0, 0, 255)
                                                                  : cv::Scalar(0, 255, 255);
                    int text_y = std::max(16, y + h + 16);
                    cv::putText(r.img, threat_text, cv::Point(x, text_y),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, threat_color, 1);
                }

                if (threat_logger.enabled()) {
                    ThreatLogRecord rec;
                    rec.frame_index = r.frame_index;
                    rec.track_id = t.track_id;
                    rec.label = t.label;
                    rec.x = x; rec.y = y; rec.w = w; rec.h = h;
                    rec.distance_m = result.distance_m;
                    rec.speed_kmh = result.speed_kmh;
                    rec.threat_score = result.threat_score;
                    rec.is_dangerous = result.is_dangerous;
                    rec.type = result.type;
                    threat_logger.log(rec);
                }
            }
            threat_analyzer.purge_stale(r.frame_index, video_fps * 2);
        } else if (opts.draw_result) {
            // 仅绘制检测框（无跟踪）
            for (const auto &o : r.objs) {
                cv::Scalar color = tracker.get_color(o.label);
                const char *label_name = get_label_name(o.label);
                char text[128];
                snprintf(text, sizeof(text), "%s %.1f%%", label_name, o.prob * 100.0f);
                cv::rectangle(r.img, o.rect, color, 2);
                cv::putText(r.img, text, cv::Point((int)o.rect.x, (int)o.rect.y - 6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
            }
        }

        // 4) 送编码线程
        if (enable_writer) {
            if (encode_failed.load()) {
                ERROR_LOG("Encoder thread failed, stopping");
                break;
            }
            push_encode(r.img);
        }

        frames++;

        // STATS 聚合
        acc_pre   += r.timing.pre_ms;
        acc_infer += r.timing.infer_ms;
        acc_post  += r.timing.post_ms;
        if (++stat_frames % 60 == 0) {
            printf("[STATS] pre=%.2fms infer=%.2fms post=%.2fms (单帧延迟，非吞吐)\n",
                   acc_pre / 60.0, acc_infer / 60.0, acc_post / 60.0);
            acc_pre = acc_infer = acc_post = 0.0;
        }

        if (opts.max_frames > 0 && frames >= opts.max_frames) {
            break;
        }

        if (frames % 60 == 0) {
            gettimeofday(&time_now, nullptr);
            tmpTime = time_now.tv_sec * 1000 + time_now.tv_usec / 1000;
            printf("[FPS] 60帧平均: %.2f FPS\n", 60000.0 / (float)(tmpTime - lopTime));
            lopTime = tmpTime;
        }
    }

    // ---- 收尾：停采集 → 排空在途推理 → 停编码 ----
    stop_requested.store(true);
    queue_not_full.notify_all();
    queue_not_empty.notify_all();
    if (capture_thread.joinable()) {
        capture_thread.join();
    }

    // 排空仍在途的推理 future（避免析构时阻塞/丢结果）
    while (!futs.empty()) {
        futs.front().get();
        futs.pop();
    }

    if (enable_writer) {
        encode_done_input.store(true);
        enc_not_empty.notify_all();
        enc_not_full.notify_all();
        if (encode_thread.joinable()) {
            encode_thread.join();
        }
    }

    gettimeofday(&time_now, nullptr);
    long totalTime = time_now.tv_sec * 1000 + time_now.tv_usec / 1000 - initTime;
    float total_fps = (totalTime > 0) ? (float)frames / (float)totalTime * 1000.0f : 0;
    printf("\n[SUMMARY] 总帧数: %d, 总耗时: %.2fs, 平均帧率: %.2f FPS\n",
           frames, totalTime / 1000.0f, total_fps);

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
