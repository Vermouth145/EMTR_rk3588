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
            "\"distance_m\":%.3f,\"speed_kmh\":%.3f,\"threat_score\":%.3f,"
            "\"dangerous\":%s,\"type\":\"%s\"}\n",
            record.frame_index, record.track_id, record.label,
            record.x, record.y, record.w, record.h,
            record.distance_m, record.speed_kmh, record.threat_score,
            record.is_dangerous ? "true" : "false", record.type);
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

    const double cx = box.x + box.w / 2.0;
    const double cy = box.y + box.h / 2.0;
    const double dist = estimate_distance_m(label, box.h);

    TrackState &st = states_[track_id];   // 不存在则插入默认值
    if (st.first_frame < 0) {
        st.first_frame = frame_index;
    }
    st.last_seen_frame = frame_index;

    // ---- 测速：滑动窗口 + 像素位移折算（对齐 video_alarm.py）----
    double speed_kmh = st.last_speed_kmh;
    double dist_window_ago = -1.0;        // 窗口前的距离，用于判断是否在接近
    if (dist > 0.0) {
        st.history.push_back(HistPoint{cx, cy, dist});
        if (static_cast<int>(st.history.size()) > speed_window_) {
            const HistPoint old = st.history.front();   // speed_window 帧之前的点
            dist_window_ago = old.dist;
            // 像素位移 → 物理位移（米）：乘以 dist/focal
            const double dx = (cx - old.cx) * (dist / focal_ref_);
            const double dy = (cy - old.cy) * (dist / focal_ref_);
            const double move_m = std::sqrt(dx * dx + dy * dy);
            const double dt = static_cast<double>(speed_window_) / fps_;
            if (dt > 0.0) {
                speed_kmh = (move_m / dt) * 3.6;
            }
            st.history.pop_front();
        }
    }
    st.last_speed_kmh = speed_kmh;

    const int age_frames = frame_index - st.first_frame + 1;
    const double dwell_s = static_cast<double>(age_frames) / fps_;

    // ---- 威胁度 + 危险判定（脚本逻辑）----
    const double score = compute_threat_score(dist, speed_kmh);
    const bool danger = (dist > 0.0 && dist < warning_range_ && score > threat_threshold_);

    // ---- 行为类型（保留三种，按优先级判定）----
    const bool approaching = (dist_window_ago > 0.0 && dist > 0.0 && dist < dist_window_ago);
    const bool fast_approach = approaching && speed_kmh > fast_approach_speed_kmh_ &&
                               dist > 0.0 && dist < warning_range_;
    const bool sudden_appear = age_frames <= sudden_age_frames_ && dist > 0.0 && dist < sudden_near_m_;
    const bool loitering = dwell_s > loiter_dwell_s_ && speed_kmh < loiter_speed_kmh_;

    if (fast_approach) {
        result.type = "fast_approach";
    } else if (sudden_appear) {
        result.type = "sudden_appear";
    } else if (loitering) {
        result.type = "loitering";
    } else {
        result.type = "normal";
    }

    result.threat_score = score;
    result.is_dangerous = danger;
    result.distance_m = dist;
    result.speed_kmh = speed_kmh;
    result.dwell_s = dwell_s;
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

double ThreatAnalyzer::compute_threat_score(double distance_m, double speed_kmh) const
{
    if (distance_m <= 0.0) {
        return 0.0;
    }
    // s_dist = max(0, 1 - dist/warning_range)
    double s_dist = 1.0 - distance_m / warning_range_;
    s_dist = std::max(0.0, std::min(1.0, s_dist));
    // s_speed = min(1, speed/max_speed)
    double s_speed = std::max(0.0, std::min(1.0, speed_kmh / max_speed_kmh_));
    return w_dist_ * s_dist + w_speed_ * s_speed;
}

double ThreatAnalyzer::estimate_distance_m(int label, int bbox_height) const
{
    if (bbox_height < min_bbox_h_) {
        return -1.0;
    }
    const double height_m = class_height_m(label);
    return (height_m * focal_ref_) / static_cast<double>(std::max(bbox_height, 1));
}

double ThreatAnalyzer::class_height_m(int label) const
{
    // EMTR 模型标签顺序（见 model/EMTR_labels_list.txt）：0=car, 1=person, 2=animal
    // 对应 video_alarm.py 中固定的 REAL_HEIGHT(1.7)，此处按类别泛化。
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
