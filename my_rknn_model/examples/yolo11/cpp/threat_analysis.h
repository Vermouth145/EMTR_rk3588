#pragma once

#include <string>
#include <unordered_map>

struct ThreatResult {
    const char *type = "normal";  // points to a string literal, no heap allocation
    int score = 0;
    double distance_m = -1.0;
    double speed_mps = 0.0;
    double dwell_s = 0.0;
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
    double speed_mps = 0.0;
    int score = 0;
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
    struct TrackState {
        int first_frame = -1;
        int last_frame = -1;
        int last_seen_frame = -1;
        double last_distance = -1.0;
        double last_speed = 0.0;
        double filtered_distance = -1.0;
        double filtered_speed = 0.0;
        int last_bbox_h = 0;
        int label = -1;
    };

    double estimate_distance_m(int label, int bbox_height) const;
    double class_height_m(int label) const;
    ThreatResult compute_threat(double distance_m, double speed_mps, int age_frames) const;

    std::unordered_map<int, TrackState> states_;
    double fps_ = 30.0;
    double focal_px_ = 700.0;
    double dist_alpha_ = 0.2;
    double speed_alpha_ = 0.2;
    int min_bbox_h_ = 20;
    double max_speed_mps_ = 10.0;
};

