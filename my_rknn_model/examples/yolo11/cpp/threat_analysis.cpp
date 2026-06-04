#include "threat_analysis.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
            "\"distance_m\":%.3f,\"speed_mps\":%.3f,\"speed_kmh\":%.3f,\"threat_score\":%.2f,"
            "\"dangerous\":%s,\"type\":\"%s\"}\n",
            record.frame_index, record.track_id, record.label,
            record.x, record.y, record.w, record.h,
            record.distance_m, record.speed_mps, record.speed_kmh, record.threat_score,
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

void ThreatAnalyzer::set_image_size(int width, int height)
{
    if (width > 0)  image_w_ = width;
    if (height > 0) image_h_ = height;
}

void ThreatAnalyzer::set_camera_geometry(double height_m, double tilt_deg)
{
    if (height_m > 0.0) {
        cam_height_m_ = height_m;
    }
    // 俯仰角限制在 [-89°, 89°]，避免 tan 病态
    const double t = std::max(-89.0, std::min(89.0, tilt_deg));
    cam_tilt_rad_ = t * M_PI / 180.0;
}

void ThreatAnalyzer::set_focal_px(double focal_px)
{
    if (focal_px > 1.0) {
        focal_ref_ = focal_px;
    }
}

void ThreatAnalyzer::set_warning_range(double range_m)
{
    if (range_m > 0.0) {
        warning_range_ = range_m;
    }
}

ThreatResult ThreatAnalyzer::update(int frame_index, int track_id, int label, const ThreatBBox &box)
{
    ThreatResult result;

    // 脚点（bbox 底边中心）——地平面测距的关键参考点，比整体高度稳健、抗截断。
    const double foot_x = box.x + box.w / 2.0;
    const double foot_y = static_cast<double>(box.y + box.h);

    // 脚点是否被画面底边截断（脚出画 → 落脚点不可靠）。
    constexpr int kEdgeMargin = 2;
    const bool bottom_truncated = (box.h <= 0) ||
                                  (foot_y >= static_cast<double>(image_h_ - kEdgeMargin));

    // ---- 混合测距：优先地平面落脚点模型，截断时退回针孔高度模型 ----
    GroundPos gp;   // valid=false 表示无地平面坐标
    double dist = -1.0;
    if (!bottom_truncated) {
        gp = project_ground(foot_x, foot_y);
        if (gp.valid) {
            dist = gp.dist;
        }
    }
    if (dist <= 0.0) {
        dist = estimate_distance_height(label, box.h);   // 兜底；过小返回 -1
    }

    TrackState &st = states_[track_id];   // 不存在则插入默认值
    if (st.first_frame < 0) {
        st.first_frame = frame_index;
    }
    st.last_seen_frame = frame_index;

    // ---- 测速：滑动窗口。脚点有效时用真实地面位移(米)，否则像素位移折算 ----
    double speed_kmh = st.last_speed_kmh;
    double dist_window_ago = -1.0;        // 窗口前的距离，用于判断是否在接近
    if (dist > 0.0) {
        HistPoint hp;
        hp.cx = foot_x;
        hp.cy = foot_y;
        hp.dist = dist;
        hp.ground_valid = gp.valid;
        hp.gx = gp.X;
        hp.gz = gp.Z;
        st.history.push_back(hp);
        if (static_cast<int>(st.history.size()) > speed_window_) {
            const HistPoint old = st.history.front();   // speed_window 帧之前的点
            dist_window_ago = old.dist;
            double move_m;
            if (gp.valid && old.ground_valid) {
                // 两端都有地平面坐标：直接量俯视平面上的真实位移（米）
                const double dx = gp.X - old.gx;
                const double dz = gp.Z - old.gz;
                move_m = std::sqrt(dx * dx + dz * dz);
            } else {
                // 兜底：像素位移 → 物理位移（米），乘以 dist/focal
                const double dx = (foot_x - old.cx) * (dist / focal_ref_);
                const double dy = (foot_y - old.cy) * (dist / focal_ref_);
                move_m = std::sqrt(dx * dx + dy * dy);
            }
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
    result.speed_mps = speed_kmh / 3.6;   // TC-UT-18 要求以 m/s 标注/判据
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
    // 量程 0~100（对齐测试说明 TC-UT-20）
    return (w_dist_ * s_dist + w_speed_ * s_speed) * 100.0;
}

// 地平面落脚点测距：把脚点像素 (u,v) 投影到地平面，求相机到目标的水平距离。
//   俯角(脚点射线相对水平向下) φ = cam_tilt + atan((v - cy0) / f)
//   前向地面距离 Z = H / tan(φ)；横向偏移 X ≈ Z * (u - cx0) / f
// 脚点在地平线及以上(φ<=0)或过远(>上限)时视为无效。
ThreatAnalyzer::GroundPos ThreatAnalyzer::project_ground(double foot_x, double foot_y) const
{
    GroundPos g;
    if (focal_ref_ <= 1.0 || cam_height_m_ <= 0.0) {
        return g;
    }
    const double cx0 = image_w_ * 0.5;
    const double cy0 = image_h_ * 0.5;
    const double ang = std::atan2(foot_y - cy0, focal_ref_);  // 脚点相对光轴的下偏角
    const double phi = cam_tilt_rad_ + ang;                   // 相对水平的俯角
    constexpr double kMinPhi = 0.0087;                        // ≈0.5°，避免近地平线发散
    if (phi <= kMinPhi) {
        return g;   // 脚点在地平线及以上，不在地面上
    }
    const double Z = cam_height_m_ / std::tan(phi);
    if (Z <= 0.0 || Z > max_ground_dist_) {
        return g;
    }
    const double X = Z * (foot_x - cx0) / focal_ref_;
    g.X = X;
    g.Z = Z;
    g.dist = std::sqrt(X * X + Z * Z);
    g.valid = true;
    return g;
}

// 针孔高度模型（兜底）：dist = real_height * focal_ref / bbox_h，对齐 video_alarm.py。
double ThreatAnalyzer::estimate_distance_height(int label, int bbox_height) const
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
