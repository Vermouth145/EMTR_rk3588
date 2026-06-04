#include "threat_analysis.h"

#include <stdio.h>

int main()
{
    ThreatAnalyzer analyzer(30.0);
    ThreatLogger logger;
    logger.open("threat_demo.jsonl");

    for (int frame = 1; frame <= 60; ++frame) {
        ThreatBBox box;
        box.x = 100;
        box.y = 100;
        box.w = 80;
        box.h = 200 - frame;  // simulate approach

        ThreatResult result = analyzer.update(frame, 1, 0, box);
        ThreatLogRecord rec;
        rec.frame_index = frame;
        rec.track_id = 1;
        rec.label = 0;
        rec.x = box.x;
        rec.y = box.y;
        rec.w = box.w;
        rec.h = box.h;
        rec.distance_m = result.distance_m;
        rec.speed_mps = result.speed_mps;
        rec.speed_kmh = result.speed_kmh;
        rec.threat_score = result.threat_score;
        rec.is_dangerous = result.is_dangerous;
        rec.type = result.type;
        logger.log(rec);

        if (frame % 10 == 0) {
            printf("frame=%d dist=%.2f speed=%.2fm/s threat=%.1f danger=%d type=%s\n",
                   frame, result.distance_m, result.speed_mps, result.threat_score,
                   result.is_dangerous ? 1 : 0, result.type);
        }
    }

    logger.close();
    printf("Wrote threat_demo.jsonl\n");
    return 0;
}

