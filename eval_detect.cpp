#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>

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
};

// 计算两个边界框之间的IoU
float calculateIoU(const Detection& det, const GroundTruth& gt) {
    float inter_x1 = std::max(det.x1, gt.x1);
    float inter_y1 = std::max(det.y1, gt.y1);
    float inter_x2 = std::min(det.x2, gt.x2);
    float inter_y2 = std::min(det.y2, gt.y2);
    
    if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) {
        return 0.0f;
    }
    
    float intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    float det_area = (det.x2 - det.x1) * (det.y2 - det.y1);
    float gt_area = (gt.x2 - gt.x1) * (gt.y2 - gt.y2);
    float union_area = det_area + gt_area - intersection_area;
    
    return intersection_area / union_area;
}

// 计算IoU阈值为0.5时的平均精度(AP50)
float calculateAP50(const std::vector<Detection>& detections,
                   const std::vector<GroundTruth>& ground_truths,
                   int class_id) {
    
    // 为特定类别过滤检测结果和真实标注
    std::vector<Detection> class_detections;
    std::vector<GroundTruth> class_ground_truths;
    
    for (const auto& det : detections) {
        if (det.class_id == class_id) {
            class_detections.push_back(det);
        }
    }
    
    for (auto gt : ground_truths) {
        if (gt.class_id == class_id) {
            class_ground_truths.push_back(gt);
        }
    }
    
    // 按置信度降序排序检测结果
    std::sort(class_detections.begin(), class_detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    if (class_ground_truths.empty()) {
        // 如果此类别没有真实标注，则当有检测结果时AP为0
        return class_detections.empty() ? 1.0f : 0.0f;
    }
    
    std::vector<bool> tp(class_detections.size(), false);  // 真正例
    std::vector<bool> fp(class_detections.size(), false);  // 假正例
    
    for (size_t i = 0; i < class_detections.size(); i++) {
        bool found_match = false;
        
        for (size_t j = 0; j < class_ground_truths.size(); j++) {
            if (!class_ground_truths[j].used && 
                calculateIoU(class_detections[i], class_ground_truths[j]) >= 0.5f) {
                
                class_ground_truths[j].used = true;  // 标记为已使用
                tp[i] = true;
                found_match = true;
                break;
            }
        }
        
        if (!found_match) {
            fp[i] = true;
        }
    }
    
    // 计算累积TP和FP
    std::vector<int> cum_tp(class_detections.size());
    std::vector<int> cum_fp(class_detections.size());
    
    cum_tp[0] = tp[0] ? 1 : 0;
    cum_fp[0] = fp[0] ? 1 : 0;
    
    for (size_t i = 1; i < class_detections.size(); i++) {
        cum_tp[i] = cum_tp[i-1] + (tp[i] ? 1 : 0);
        cum_fp[i] = cum_fp[i-1] + (fp[i] ? 1 : 0);
    }
    
    // 计算精确率和召回率
    std::vector<float> precisions(class_detections.size());
    std::vector<float> recalls(class_detections.size());
    
    for (size_t i = 0; i < class_detections.size(); i++) {
        int total_positives = cum_tp[i] + cum_fp[i];
        if (total_positives == 0) {
            precisions[i] = 0;
        } else {
            precisions[i] = static_cast<float>(cum_tp[i]) / total_positives;
        }
        recalls[i] = static_cast<float>(cum_tp[i]) / class_ground_truths.size();
    }
    
    // 应用11点插值法计算AP
    float ap = 0.0f;
    for (float r = 0.0f; r <= 1.0f; r += 0.1f) {
        float max_precision = 0.0f;
        for (size_t i = 0; i < recalls.size(); i++) {
            if (recalls[i] >= r && precisions[i] > max_precision) {
                max_precision = precisions[i];
            }
        }
        ap += max_precision;
    }
    ap /= 11.0f;
    
    // 重置标记位以供下次类别评估使用
    for (auto& gt : class_ground_truths) {
        gt.used = false;
    }
    
    return ap;
}

// 计算特定类别的精确率和召回率
std::pair<float, float> calculatePrecisionRecall(const std::vector<Detection>& detections,
                                               const std::vector<GroundTruth>& ground_truths,
                                               int class_id) {
    
    // 为特定类别过滤检测结果和真实标注
    std::vector<Detection> class_detections;
    std::vector<GroundTruth> class_ground_truths;
    
    for (const auto& det : detections) {
        if (det.class_id == class_id) {
            class_detections.push_back(det);
        }
    }
    
    for (auto gt : ground_truths) {
        if (gt.class_id == class_id) {
            class_ground_truths.push_back(gt);
        }
    }
    
    // 按置信度降序排序检测结果
    std::sort(class_detections.begin(), class_detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    if (class_detections.empty() && class_ground_truths.empty()) {
        return {1.0f, 1.0f};  // 无检测目标时的完美情况
    }
    
    if (class_ground_truths.empty()) {
        return {0.0f, 0.0f};  // 有检测但无真实标注 -> 精确率为0
    }
    
    std::vector<bool> tp(class_detections.size(), false);  // 真正例
    std::vector<bool> fp(class_detections.size(), false);  // 假正例
    
    for (size_t i = 0; i < class_detections.size(); i++) {
        bool found_match = false;
        
        for (size_t j = 0; j < class_ground_truths.size(); j++) {
            if (!class_ground_truths[j].used && 
                calculateIoU(class_detections[i], class_ground_truths[j]) >= 0.5f) {
                
                class_ground_truths[j].used = true;  // 标记为已使用
                tp[i] = true;
                found_match = true;
                break;
            }
        }
        
        if (!found_match) {
            fp[i] = true;
        }
    }
    
    // 统计最终的TP和FP
    int total_tp = 0;
    int total_fp = 0;
    
    for (bool t : tp) {
        if (t) total_tp++;
    }
    for (bool f : fp) {
        if (f) total_fp++;
    }
    
    // 计算精确率和召回率
    float precision = (total_tp + total_fp == 0) ? 0.0f : static_cast<float>(total_tp) / (total_tp + total_fp);
    float recall = static_cast<float>(total_tp) / class_ground_truths.size();
    
    // 重置标记位以供下次类别评估使用
    for (auto& gt : class_ground_truths) {
        gt.used = false;
    }
    
    return {precision, recall};
}

// 计算所有类别的mAP50
float calculateMAP50(const std::vector<Detection>& detections,
                    const std::vector<GroundTruth>& ground_truths) {
    
    // 获取唯一的类别ID
    std::vector<int> class_ids;
    for (const auto& gt : ground_truths) {
        if (std::find(class_ids.begin(), class_ids.end(), gt.class_id) == class_ids.end()) {
            class_ids.push_back(gt.class_id);
        }
    }
    
    float total_ap = 0.0f;
    int valid_classes = 0;
    
    for (int class_id : class_ids) {
        float ap = calculateAP50(detections, ground_truths, class_id);
        total_ap += ap;
        valid_classes++;
    }
    
    return valid_classes > 0 ? total_ap / valid_classes : 0.0f;
}

// 计算所有类别的平均精确率和召回率
std::pair<float, float> calculateAvgPrecisionRecall(const std::vector<Detection>& detections,
                                                  const std::vector<GroundTruth>& ground_truths) {
    
    // 获取唯一的类别ID
    std::vector<int> class_ids;
    for (const auto& gt : ground_truths) {
        if (std::find(class_ids.begin(), class_ids.end(), gt.class_id) == class_ids.end()) {
            class_ids.push_back(gt.class_id);
        }
    }
    
    float total_precision = 0.0f;
    float total_recall = 0.0f;
    int valid_classes = 0;
    
    for (int class_id : class_ids) {
        auto pr = calculatePrecisionRecall(detections, ground_truths, class_id);
        total_precision += pr.first;
        total_recall += pr.second;
        valid_classes++;
    }
    
    if (valid_classes > 0) {
        return {total_precision / valid_classes, total_recall / valid_classes};
    } else {
        return {0.0f, 0.0f};
    }
}

// 结构体，存储单个类别的详细指标
struct ClassMetrics {
    float precision;   // 精确率
    float recall;      // 召回率
    float ap50;        // AP50
    int tp_count;      // 真正例数量
    int fp_count;      // 假正例数量
    int fn_count;      // 假负例数量
};

// 计算每个类别的详细指标
std::map<int, ClassMetrics> calculateDetailedMetrics(const std::vector<Detection>& detections,
                                                   const std::vector<GroundTruth>& ground_truths) {
    
    std::map<int, ClassMetrics> metrics;
    
    // 获取唯一的类别ID
    std::vector<int> class_ids;
    for (const auto& gt : ground_truths) {
        if (std::find(class_ids.begin(), class_ids.end(), gt.class_id) == class_ids.end()) {
            class_ids.push_back(gt.class_id);
        }
    }
    
    for (int class_id : class_ids) {
        // 为特定类别过滤检测结果和真实标注
        std::vector<Detection> class_detections;
        std::vector<GroundTruth> class_ground_truths;
        
        for (const auto& det : detections) {
            if (det.class_id == class_id) {
                class_detections.push_back(det);
            }
        }
        
        for (auto gt : ground_truths) {
            if (gt.class_id == class_id) {
                class_ground_truths.push_back(gt);
            }
        }
        
        // 按置信度降序排序检测结果
        std::sort(class_detections.begin(), class_detections.end(),
                  [](const Detection& a, const Detection& b) {
                      return a.confidence > b.confidence;
                  });
        
        // 初始化计数
        int tp_count = 0;
        int fp_count = 0;
        
        // 重置标记位
        for (auto& gt : class_ground_truths) {
            gt.used = false;
        }
        
        // 计算TP和FP
        for (size_t i = 0; i < class_detections.size(); i++) {
            bool found_match = false;
            
            for (size_t j = 0; j < class_ground_truths.size(); j++) {
                if (!class_ground_truths[j].used && 
                    calculateIoU(class_detections[i], class_ground_truths[j]) >= 0.5f) {
                    
                    class_ground_truths[j].used = true;  // 标记为已使用
                    tp_count++;
                    found_match = true;
                    break;
                }
            }
            
            if (!found_match) {
                fp_count++;
            }
        }
        
        // 计算FN（未匹配的真实标注数量）
        int fn_count = 0;
        for (const auto& gt : class_ground_truths) {
            if (!gt.used) {
                fn_count++;
            }
        }
        
        // 计算精确率和召回率
        float precision = (tp_count == 0) ? 0.0f : static_cast<float>(tp_count) / (tp_count + fp_count);
        float recall = (tp_count == 0) ? 0.0f : static_cast<float>(tp_count) / (tp_count + fn_count);
        
        // 计算此类别的AP50
        float ap50 = calculateAP50(detections, ground_truths, class_id);
        
        // 存储指标
        metrics[class_id] = {precision, recall, ap50, tp_count, fp_count, fn_count};
    }
    
    return metrics;
}