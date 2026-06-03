# RK3588 YOLO11 Demo Extensions

This folder extends the YOLO11 demo with:
- FFmpeg `h264_rkmpp` encoding to MP4/AVI.
- Per-track distance/speed estimates.
- Threat scoring with JSONL logs.

## Quick Run (board)

````bash
./rknn_yolo11_demo model/yolo11.rknn --input /userdata/test.mp4 --output /userdata/out.mp4 --codec h264_rkmpp --json /userdata/threat.jsonl
````

## Options

- `--codec h264_rkmpp`: use Rockchip MPP encoder via FFmpeg.
- `--json <path>`: enable JSONL log output.
- `--no-json`: disable JSON logging.

## Threat Log Format (JSONL)

Each line is one frame per track:

````json
{"frame":12,"track_id":3,"label":0,"bbox":[100,80,60,160],"distance_m":6.42,"speed_mps":1.82,"score":80,"type":"fast_approach"}
````

Notes:
- Distance/speed are approximate (pinhole model with default class heights).
- Update `ThreatAnalyzer::class_height_m` for custom class sizes.

