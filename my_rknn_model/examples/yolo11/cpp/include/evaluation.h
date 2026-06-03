#ifndef EVALUATION_H
#define EVALUATION_H

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// 检测结果结构体
struct Detection {
    float x1, y1, x2, y2;  // 边界框坐标
    float confidence;       // 置信度
    int class_id;          // 类别ID
    
    Detection(float x1, float y1, float x2, float y2, float conf, int cls) 
        : x1(x1), y1(y1), x2(x2), y2(y2), confidence(conf), class_id(cls) {}
};

// 真实标注结构体
struct GroundTruth {
    float x1, y1, x2, y2;  // 边界框坐标
    int class_id;          // 类别ID
    bool used;             // 标记此真实标注是否已被匹配
    
    GroundTruth(float x1, float y1, float x2, float y2, int cls) 
        : x1(x1), y1(y1), x2(x2), y2(y2), class_id(cls), used(false) {}
    
    GroundTruth() : x1(0), y1(0), x2(0), y2(0), class_id(0), used(false) {}
};

// 结构体，存储单个类别的详细指标
struct ClassMetrics {
    float precision;   // 精确率
    float recall;      // 召回率
    float ap50;        // AP50
    int tp_count;      // 真正例数量
    int fp_count;      // 假正例数量
    int fn_count;      // 假负例数量
};

// 函数声明
float calculateIoU(const Detection& det, const GroundTruth& gt);
float calculateAP50(const std::vector<Detection>& detections,
                   const std::vector<GroundTruth>& ground_truths,
                   int class_id);
float calculateMAP50(const std::vector<Detection>& detections,
                    const std::vector<GroundTruth>& ground_truths);
std::pair<float, float> calculateAvgPrecisionRecall(const std::vector<Detection>& detections,
                                                  const std::vector<GroundTruth>& ground_truths);
std::map<int, ClassMetrics> calculateDetailedMetrics(const std::vector<Detection>& detections,
                                                   const std::vector<GroundTruth>& ground_truths);
void printMetrics(const std::map<int, ClassMetrics>& metrics);
void saveMetricsToFile(const std::map<int, ClassMetrics>& metrics, const std::string& filename);
std::vector<GroundTruth> loadGroundTruthFromFile(const std::string& filename, float scale_x, float scale_y);
std::vector<GroundTruth> loadGroundTruthFromTxt(const std::string& filename);

#endif
