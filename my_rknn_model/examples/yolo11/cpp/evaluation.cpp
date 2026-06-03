#include "evaluation.h"
#include <algorithm>
#include <cmath>
#include <iomanip>

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
    float gt_area = (gt.x2 - gt.x1) * (gt.y2 - gt.y1);
    float union_area = det_area + gt_area - intersection_area;
    
    return intersection_area / union_area;
}

// 计算IoU阈值为0.5时的平均精度(AP50)
float calculateAP50(const std::vector<Detection>& detections,
                   const std::vector<GroundTruth>& ground_truths,
                   int class_id) {
    
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
    
    std::sort(class_detections.begin(), class_detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    if (class_ground_truths.empty()) {
        return class_detections.empty() ? 1.0f : 0.0f;
    }
    
    std::vector<bool> tp(class_detections.size(), false);
    std::vector<bool> fp(class_detections.size(), false);
    
    for (size_t i = 0; i < class_detections.size(); i++) {
        bool found_match = false;
        
        for (size_t j = 0; j < class_ground_truths.size(); j++) {
            if (!class_ground_truths[j].used && 
                calculateIoU(class_detections[i], class_ground_truths[j]) >= 0.5f) {
                
                class_ground_truths[j].used = true;
                tp[i] = true;
                found_match = true;
                break;
            }
        }
        
        if (!found_match) {
            fp[i] = true;
        }
    }
    
    std::vector<int> cum_tp(class_detections.size());
    std::vector<int> cum_fp(class_detections.size());
    
    cum_tp[0] = tp[0] ? 1 : 0;
    cum_fp[0] = fp[0] ? 1 : 0;
    
    for (size_t i = 1; i < class_detections.size(); i++) {
        cum_tp[i] = cum_tp[i-1] + (tp[i] ? 1 : 0);
        cum_fp[i] = cum_fp[i-1] + (fp[i] ? 1 : 0);
    }
    
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
    
    for (auto& gt : class_ground_truths) {
        gt.used = false;
    }
    
    return ap;
}

// 计算所有类别的mAP50
float calculateMAP50(const std::vector<Detection>& detections,
                    const std::vector<GroundTruth>& ground_truths) {
    
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
        
        // 重置标记位
        for (auto& gt : class_ground_truths) {
            gt.used = false;
        }
        
        // 计算TP和FP
        int tp_count = 0;
        int fp_count = 0;
        
        for (size_t i = 0; i < class_detections.size(); i++) {
            bool found_match = false;
            
            for (size_t j = 0; j < class_ground_truths.size(); j++) {
                if (!class_ground_truths[j].used && 
                    calculateIoU(class_detections[i], class_ground_truths[j]) >= 0.5f) {
                    
                    class_ground_truths[j].used = true;
                    tp_count++;
                    found_match = true;
                    break;
                }
            }
            
            if (!found_match) {
                fp_count++;
            }
        }
        
        // 计算FN
        int fn_count = 0;
        for (const auto& gt : class_ground_truths) {
            if (!gt.used) {
                fn_count++;
            }
        }
        
        float precision = (tp_count == 0) ? 0.0f : static_cast<float>(tp_count) / (tp_count + fp_count);
        float recall = (tp_count == 0) ? 0.0f : static_cast<float>(tp_count) / (tp_count + fn_count);
        
        total_precision += precision;
        total_recall += recall;
        valid_classes++;
    }
    
    if (valid_classes > 0) {
        return {total_precision / valid_classes, total_recall / valid_classes};
    } else {
        return {0.0f, 0.0f};
    }
}

// 计算每个类别的详细指标
std::map<int, ClassMetrics> calculateDetailedMetrics(const std::vector<Detection>& detections,
                                                   const std::vector<GroundTruth>& ground_truths) {
    
    std::map<int, ClassMetrics> metrics;
    
    std::vector<int> class_ids;
    for (const auto& gt : ground_truths) {
        if (std::find(class_ids.begin(), class_ids.end(), gt.class_id) == class_ids.end()) {
            class_ids.push_back(gt.class_id);
        }
    }
    
    for (int class_id : class_ids) {
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
        
        std::sort(class_detections.begin(), class_detections.end(),
                  [](const Detection& a, const Detection& b) {
                      return a.confidence > b.confidence;
                  });
        
        int tp_count = 0;
        int fp_count = 0;
        
        for (auto& gt : class_ground_truths) {
            gt.used = false;
        }
        
        for (size_t i = 0; i < class_detections.size(); i++) {
            bool found_match = false;
            
            for (size_t j = 0; j < class_ground_truths.size(); j++) {
                if (!class_ground_truths[j].used && 
                    calculateIoU(class_detections[i], class_ground_truths[j]) >= 0.5f) {
                    
                    class_ground_truths[j].used = true;
                    tp_count++;
                    found_match = true;
                    break;
                }
            }
            
            if (!found_match) {
                fp_count++;
            }
        }
        
        int fn_count = 0;
        for (const auto& gt : class_ground_truths) {
            if (!gt.used) {
                fn_count++;
            }
        }
        
        float precision = (tp_count == 0) ? 0.0f : static_cast<float>(tp_count) / (tp_count + fp_count);
        float recall = (tp_count == 0) ? 0.0f : static_cast<float>(tp_count) / (tp_count + fn_count);
        float ap50 = calculateAP50(detections, ground_truths, class_id);
        
        metrics[class_id] = {precision, recall, ap50, tp_count, fp_count, fn_count};
    }
    
    return metrics;
}

// 打印详细指标
void printMetrics(const std::map<int, ClassMetrics>& metrics) {
    std::cout << "\n========== 评估指标 ==========" << std::endl;
    std::cout << std::left << std::setw(10) << "Class" 
              << std::setw(12) << "Precision" 
              << std::setw(12) << "Recall" 
              << std::setw(12) << "AP50"
              << std::setw(10) << "TP"
              << std::setw(10) << "FP"
              << std::setw(10) << "FN" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    for (const auto& [class_id, m] : metrics) {
        std::cout << std::left << std::setw(10) << class_id
                  << std::setw(12) << std::fixed << std::setprecision(4) << m.precision
                  << std::setw(12) << m.recall
                  << std::setw(12) << m.ap50
                  << std::setw(10) << m.tp_count
                  << std::setw(10) << m.fp_count
                  << std::setw(10) << m.fn_count << std::endl;
    }
    std::cout << "========================================" << std::endl;
}

// 保存指标到文件
void saveMetricsToFile(const std::map<int, ClassMetrics>& metrics, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    
    file << "Class,Precision,Recall,AP50,TP,FP,FN" << std::endl;
    
    for (const auto& [class_id, m] : metrics) {
        file << class_id << ","
             << m.precision << ","
             << m.recall << ","
             << m.ap50 << ","
             << m.tp_count << ","
             << m.fp_count << ","
             << m.fn_count << std::endl;
    }
    
    file.close();
    std::cout << "指标已保存到: " << filename << std::endl;
}

// 从文件加载真实标注（YOLO格式）
std::vector<GroundTruth> loadGroundTruthFromFile(const std::string& filename, float scale_x, float scale_y) {
    std::vector<GroundTruth> ground_truths;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法打开标注文件: " << filename << std::endl;
        return ground_truths;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int class_id;
        float cx, cy, w, h;
        
        if (iss >> class_id >> cx >> cy >> w >> h) {
            // 将归一化坐标转换为像素坐标
            float x1 = (cx - w/2) * scale_x;
            float y1 = (cy - h/2) * scale_y;
            float x2 = (cx + w/2) * scale_x;
            float y2 = (cy + h/2) * scale_y;
            
            ground_truths.emplace_back(x1, y1, x2, y2, class_id);
        }
    }
    
    file.close();
    return ground_truths;
}

// 从TXT文件加载真实标注（绝对坐标格式）
std::vector<GroundTruth> loadGroundTruthFromTxt(const std::string& filename) {
    std::vector<GroundTruth> ground_truths;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法打开标注文件: " << filename << std::endl;
        return ground_truths;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int class_id;
        float x1, y1, x2, y2;
        
        if (iss >> class_id >> x1 >> y1 >> x2 >> y2) {
            ground_truths.emplace_back(x1, y1, x2, y2, class_id);
        }
    }
    
    file.close();
    return ground_truths;
}
