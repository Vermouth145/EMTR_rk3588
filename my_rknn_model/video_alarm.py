import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# 强制禁用显示
os.environ["DISPLAY"] = ""
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- 预警参数配置 ---
REAL_HEIGHT = 1.7  # 行人平均高度(米)
FOCAL_REF = 1200  # 焦距参考值
WARNING_RANGE = 15.0  # 预警范围：15米以内
THREAT_THRESHOLD = 0.6  # 威胁度阈值 (0.0 - 1.0)
W_DIST = 0.6  # 距离权重
W_SPEED = 0.4  # 速度权重


class ThreatAnalyzer:
    def __init__(self):
        self.track_history = defaultdict(lambda: [])
        self.last_speed = defaultdict(lambda: 0.0)

    def calculate_threat(self, dist, speed):
        """计算威胁度数值 (0.0 - 1.0)"""
        s_dist = max(0, 1 - (dist / WARNING_RANGE))
        max_speed = 20.0
        s_speed = min(1.0, speed / max_speed)
        threat_score = (W_DIST * s_dist) + (W_SPEED * s_speed)
        return threat_score

    def process_frame(self, frame, results, fps):
        # --- 修复后的判断逻辑 ---
        if results[0].boxes is None or results[0].boxes.id is None:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()

        # 确保 fps 有效
        current_fps = fps if fps > 0 else 30

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            h_px = y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # 1. 测距
            dist = (REAL_HEIGHT * FOCAL_REF) / max(h_px, 1)  # 防止除零

            # 2. 测速 (滑动窗口平滑)
            self.track_history[track_id].append((cx, cy, dist))
            if len(self.track_history[track_id]) > 8:
                old_x, old_y, _ = self.track_history[track_id][-8]
                # 估算物理位移 (单位换算参考)
                dx = (cx - old_x) * (dist / FOCAL_REF)
                dy = (cy - old_y) * (dist / FOCAL_REF)
                move_dist = np.sqrt(dx ** 2 + dy ** 2)
                speed_kmh = (move_dist / (8 / current_fps)) * 3.6
                self.last_speed[track_id] = speed_kmh
                self.track_history[track_id].pop(0)

            current_speed = self.last_speed[track_id]

            # 3. 威胁度判定
            threat_score = self.calculate_threat(dist, current_speed)

            # 颜色逻辑
            is_dangerous = dist < WARNING_RANGE and threat_score > THREAT_THRESHOLD
            color = (0, 0, 255) if is_dangerous else (0, 255, 0)
            label_prefix = "!!! DANGER !!!" if is_dangerous else "SAFE"

            # 4. 绘图
            display_msg = f"{label_prefix} ID:{track_id} | {dist:.1f}m | {current_speed:.1f}km/h | T:{threat_score:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, display_msg, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

        return frame


def main():
    model = YOLO("/data/qinhui/codes/yolov11/runs/detect/LLVIP_split0-7_train_8_test_rgb/weights/best.pt")
    input_video = "/data/qinhui/data/LLVIP/LLVIP_raw_videos/visible/05.mp4"
    output_video = "/data/qinhui/data/LLVIP/LLVIP_raw_videos/visible_detected/05_warning_final.mp4"

    cap = cv2.VideoCapture(input_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    analyzer = ThreatAnalyzer()

    print(f"开始处理视频: {input_video} (FPS: {fps})")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # 运行跟踪
        results = model.track(frame, persist=True, verbose=False)

        # 处理预警
        frame = analyzer.process_frame(frame, results, fps)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"处理完成！输出文件：{output_video}")


if __name__ == "__main__":
    main()