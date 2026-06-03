#include "threat_analysis.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

ThreatLogger::ThreatLogger() = default;

ThreatLogger::~ThreatLogger()
{
    close();
}

bool ThreatLogger::open(const std::string &path)
{
    close();
    path_ = path;
    fp_ = fopen(path.c_str(), "w");
    return fp_ != nullptr;
}

void ThreatLogger::close()
{
    if (fp_) {
        fclose(fp_);
        fp_ = nullptr;
    }
}

bool ThreatLogger::enabled() const
{
    return fp_ != nullptr;
}

void ThreatLogger::log(const ThreatLogRecord &record)
{
    if (!fp_) {
        return;
    }
    fprintf(fp_,
            "{\"frame\":%d,\"track_id\":%d,\"label\":%d,\"bbox\":[%d,%d,%d,%d],"
            "\"distance_m\":%.3f,\"speed_mps\":%.3f,\"score\":%d,\"type\":\"%s\"}\n",
            record.frame_index, record.track_id, record.label,
            record.x, record.y, record.w, record.h,
            record.distance_m, record.speed_mps, record.score, record.type);
    if (++unflushed_ >= kFlushInterval) {
        fflush(fp_);
        unflushed_ = 0;
    }
}

void ThreatLogger::flush()
{
    if (fp_) {
        fflush(fp_);
        unflushed_ = 0;
    }
}

ThreatAnalyzer::ThreatAnalyzer(double fps)
{
    set_fps(fps);
}

void ThreatAnalyzer::set_fps(double fps)
{
    if (fps > 1.0) {
        fps_ = fps;
    }
}

ThreatResult ThreatAnalyzer::update(int frame_index, int track_id, int label, const ThreatBBox &box)
{
    ThreatResult result;

    double raw_distance = estimate_distance_m(label, box.h);

    auto it = states_.find(track_id);
    if (it == states_.end()) {
        TrackState state;
        state.first_frame = frame_index;
        state.last_frame = frame_index;
        state.last_seen_frame = frame_index;
        state.last_distance = raw_distance;
        state.filtered_distance = raw_distance;
        state.filtered_speed = 0.0;
        state.last_bbox_h = box.h;
        state.label = label;
        states_.emplace(track_id, state);
        result = compute_threat(raw_distance, 0.0, 1);
        result.distance_m = raw_distance;
        result.speed_mps = 0.0;
        result.dwell_s = 1.0 / fps_;
        return result;
    }

    TrackState &state = it->second;
    int frame_delta = frame_index - state.last_frame;
    double filtered_distance = state.filtered_distance;
    double filtered_speed = state.filtered_speed;

    if (box.h >= min_bbox_h_) {
        if (state.filtered_distance < 0.0) {
            filtered_distance = raw_distance;
        } else if (raw_distance > 0.0) {
            filtered_distance = state.filtered_distance * (1.0 - dist_alpha_) + raw_distance * dist_alpha_;
        }
        state.last_bbox_h = box.h;
    }

    if (frame_delta > 0 && filtered_distance > 0.0 && state.filtered_distance > 0.0) {
        double dt = static_cast<double>(frame_delta) / fps_;
        if (dt > 0.0) {
            double raw_speed = (state.filtered_distance - filtered_distance) / dt;
            raw_speed = std::max(-max_speed_mps_, std::min(max_speed_mps_, raw_speed));
            filtered_speed = state.filtered_speed * (1.0 - speed_alpha_) + raw_speed * speed_alpha_;
        }
    }

    state.last_frame = frame_index;
    state.last_seen_frame = frame_index;
    state.last_distance = raw_distance;
    state.filtered_distance = filtered_distance;
    state.filtered_speed = filtered_speed;
    state.last_speed = filtered_speed;
    state.label = label;

    int age_frames = frame_index - state.first_frame + 1;
    result = compute_threat(filtered_distance, filtered_speed, age_frames);
    result.distance_m = filtered_distance;
    result.speed_mps = filtered_speed;
    result.dwell_s = static_cast<double>(age_frames) / fps_;

    return result;
}

void ThreatAnalyzer::purge_stale(int current_frame, int max_age_frames)
{
    if (max_age_frames <= 0) {
        return;
    }
    for (auto it = states_.begin(); it != states_.end();) {
        if (current_frame - it->second.last_seen_frame > max_age_frames) {
            it = states_.erase(it);
        } else {
            ++it;
        }
    }
}

ThreatResult ThreatAnalyzer::compute_threat(double distance_m, double speed_mps, int age_frames) const
{
    ThreatResult result;

    double dwell_s = static_cast<double>(age_frames) / fps_;
    bool fast_approach = speed_mps > 2.0 && distance_m > 0.0 && distance_m < 25.0;
    bool sudden_appear = age_frames <= 5 && distance_m > 0.0 && distance_m < 8.0;
    bool loitering = dwell_s > 10.0 && std::abs(speed_mps) < 0.3;

    int score = 0;
    if (distance_m > 0.0) {
        if (distance_m < 5.0) {
            score += 20;
        } else if (distance_m < 10.0) {
            score += 10;
        }
    }
    if (fast_approach) {
        score += 60 + std::min(20, static_cast<int>(speed_mps * 5.0));
    }
    if (sudden_appear) {
        score += 35;
    }
    if (loitering) {
        score += 25;
    }
    score = std::min(100, score);

    if (fast_approach) {
        result.type = "fast_approach";
    } else if (sudden_appear) {
        result.type = "sudden_appear";
    } else if (loitering) {
        result.type = "loitering";
    } else {
        result.type = "normal";
    }
    result.score = score;
    result.dwell_s = dwell_s;
    return result;
}

double ThreatAnalyzer::estimate_distance_m(int label, int bbox_height) const
{
    if (bbox_height < min_bbox_h_) {
        return -1.0;
    }
    double height_m = class_height_m(label);
    return (height_m * focal_px_) / static_cast<double>(bbox_height);
}

double ThreatAnalyzer::class_height_m(int label) const
{
    // EMTR 模型标签顺序（见 model/EMTR_labels_list.txt）：0=car, 1=person, 2=animal
    switch (label) {
        case 0:  // car
            return 1.5;
        case 1:  // person
            return 1.7;
        case 2:  // animal
            return 0.8;
        default:
            return 1.7;
    }
}
