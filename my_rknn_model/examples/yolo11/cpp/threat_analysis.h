#pragma once

#include <string>
#include <deque>
#include <unordered_map>

// 威胁分析结果。参数与算法对齐 video_alarm.py：
//   - 测距：针孔模型  dist = real_height * focal_ref / bbox_h
//   - 测速：speed_window 帧滑动窗口，中心点像素位移按 dist/focal 折算成米，输出 km/h
//   - 威胁度：threat_score = w_dist*s_dist + w_speed*s_speed ∈ [0,1]
//   - 危险判定：dist < warning_range 且 threat_score > threshold
// 在脚本基础上额外保留三种行为类型（fast_approach / sudden_appear / loitering）。
struct ThreatResult {
    const char *type = "normal";    // 行为类型：fast_approach / sudden_appear / loitering / normal
    double threat_score = 0.0;      // 威胁度 0.0 - 1.0
    bool is_dangerous = false;      // 是否危险（dist<warning_range 且 score>threshold）
    double distance_m = -1.0;       // 估计距离（米），-1 表示无效
    double speed_kmh = 0.0;         // 估计速度（km/h）
    double dwell_s = 0.0;           // 目标存活/逗留时间（秒）
};

struct ThreatLogRecord {
    int frame_index = 0;
    int track_id = 0;
    int label = 0;
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    double distance_m = -1.0;
    double speed_kmh = 0.0;
    double threat_score = 0.0;
    bool is_dangerous = false;
    const char *type = "normal";
};

class ThreatLogger {
public:
    ThreatLogger();
    ~ThreatLogger();

    bool open(const std::string &path);
    void close();
    bool enabled() const;
    void log(const ThreatLogRecord &record);
    void flush();

private:
    std::string path_;
    FILE *fp_ = nullptr;
    int unflushed_ = 0;
    static constexpr int kFlushInterval = 30;
};

struct ThreatBBox {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

class ThreatAnalyzer {
public:
    explicit ThreatAnalyzer(double fps = 30.0);

    void set_fps(double fps);
    ThreatResult update(int frame_index, int track_id, int label, const ThreatBBox &box);
    void purge_stale(int current_frame, int max_age_frames);

private:
    struct HistPoint {
        double cx = 0.0;
        double cy = 0.0;
        double dist = 0.0;
    };

    struct TrackState {
        int first_frame = -1;
        int last_seen_frame = -1;
        std::deque<HistPoint> history;   // 最近 speed_window(+1) 个中心点 + 距离
        double last_speed_kmh = 0.0;
    };

    double estimate_distance_m(int label, int bbox_height) const;
    double class_height_m(int label) const;
    double compute_threat_score(double distance_m, double speed_kmh) const;

    std::unordered_map<int, TrackState> states_;
    double fps_ = 30.0;

    // ---- 算法参数（对齐 video_alarm.py）----
    double focal_ref_ = 1200.0;       // FOCAL_REF：焦距参考值（与拍摄分辨率/镜头相关，需按相机标定）
    double warning_range_ = 15.0;     // WARNING_RANGE：预警距离（米）
    double threat_threshold_ = 0.6;   // THREAT_THRESHOLD：危险威胁度阈值
    double w_dist_ = 0.6;             // W_DIST：距离权重
    double w_speed_ = 0.4;            // W_SPEED：速度权重
    double max_speed_kmh_ = 20.0;     // 速度归一化上限（km/h）
    int speed_window_ = 8;            // 测速滑动窗口帧数
    int min_bbox_h_ = 10;             // 参与测距的最小 bbox 高度（像素）

    // ---- 行为判定阈值（保留三种行为，单位换算到 km/h）----
    double fast_approach_speed_kmh_ = 7.0;   // 快速接近：速度阈值
    double sudden_near_m_ = 8.0;             // 突然出现：近距离阈值
    int sudden_age_frames_ = 5;              // 突然出现：最大帧龄
    double loiter_dwell_s_ = 10.0;           // 长期逗留：时间阈值
    double loiter_speed_kmh_ = 1.0;          // 长期逗留：低速阈值
};
