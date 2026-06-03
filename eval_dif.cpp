#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

class DamageAnalyzer {
public:
    /**
     * 计算前后帧差异并提取二值化图像
     * 
     */
    static cv::Mat computeFrameDifference(const cv::Mat& before_frame, 
                                        const cv::Mat& after_frame, 
                                        int threshold = 30) {
        cv::Mat gray_before, gray_after;
        
        // 转换为灰度图
        cv::cvtColor(before_frame, gray_before, cv::COLOR_BGR2GRAY);
        cv::cvtColor(after_frame, gray_after, cv::COLOR_BGR2GRAY);
        
        // 计算绝对差分
        cv::Mat diff;
        cv::absdiff(gray_before, gray_after, diff);
        
        // 二值化处理
        cv::Mat diff_binary;
        cv::threshold(diff, diff_binary, threshold, 255, cv::THRESH_BINARY);
        
        return diff_binary;
    }
    
    /**
     * 从差异图像中提取毁伤轮廓
     */
    static std::vector<std::vector<cv::Point>> extractDamageContours(
        const cv::Mat& diff_binary,
        const cv::Mat& wall_mask = cv::Mat(),
        double minAreaRatio = 0.001,
        int frameArea = 0) {
        
        cv::Mat work_diff = diff_binary.clone();
        
        // 应用墙体掩码，只保留墙体区域内的差异
        if (!wall_mask.empty()) {
            cv::bitwise_and(work_diff, wall_mask, work_diff);
        }
        
        // 查找所有轮廓
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(work_diff, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // 计算最小面积阈值
        int minArea = frameArea > 0 ? static_cast<int>(minAreaRatio * frameArea) : 500;
        
        // 过滤小轮廓和不规则轮廓
        std::vector<std::vector<cv::Point>> damageContours;
        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt);
            if (area > minArea) {
                // 过滤过于细长的轮廓(可能是噪声)
                cv::Rect boundingRect = cv::boundingRect(cnt);
                double aspectRatio = std::max(boundingRect.width, boundingRect.height) / 
                                   (std::min(boundingRect.width, boundingRect.height) + 1e-5);
                
                if (aspectRatio < 10) {  // 避免细长噪声
                    // 进一步过滤：确保轮廓在墙体区域内（如果提供了wall_mask）
                    if (!wall_mask.empty()) {
                        cv::Mat mask = cv::Mat::zeros(wall_mask.size(), CV_8UC1);
                        cv::drawContours(mask, std::vector<std::vector<cv::Point>>{cnt}, -1, 
                                       cv::Scalar(255), -1);
                        
                        int contourPixels = cv::countNonZero(mask);
                        cv::Mat maskAndWall;
                        cv::bitwise_and(mask, wall_mask, maskAndWall);
                        int maskPixels = cv::countNonZero(maskAndWall);
                        
                        // 如果轮廓大部分在墙体区域内
                        if (maskPixels > 0.7 * contourPixels) {
                            damageContours.push_back(cnt);
                        }
                    } else {
                        damageContours.push_back(cnt);
                    }
                }
            }
        }
        
        return damageContours;
    }
    
    /**
     * 计算像素到米的转换比例
     */
    static double calculatePixelToMeterRatio(double distance, 
                                          double focalLength, 
                                          double sensorHeight, 
                                          int imageHeight) {
        // 计算垂直方向上的比例
        // 公式: 实际高度 = (传感器高度 × 距离) / 焦距
        // 每像素对应的实际高度 = 实际高度 / 图像高度
        return (sensorHeight * distance) / (focalLength * imageHeight);
    }
    
    /**
     * 检测地面线位置
     */
    static int getGroundLine(const cv::Mat& image, 
                           const cv::Rect& roi = cv::Rect(0, 0, 0, 0),
                           int minLineLength = 100) {
        cv::Rect actualRoi = roi;
        cv::Mat roiImage;
        
        if (roi.area() > 0) {
            actualRoi = roi;
            roiImage = image(cv::Range(actualRoi.y, actualRoi.y + actualRoi.height),
                           cv::Range(actualRoi.x, actualRoi.x + actualRoi.width));
        } else {
            roiImage = image;
            actualRoi = cv::Rect(0, 0, image.cols, image.rows);
        }
        
        // 转换为灰度图并应用Canny边缘检测
        cv::Mat gray;
        if (roiImage.channels() == 3) {
            cv::cvtColor(roiImage, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = roiImage;
        }
        
        cv::Mat edges;
        cv::Canny(gray, edges, 50, 150, 3);
        
        // 霍夫直线变换
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, minLineLength, 10);
        
        if (lines.empty()) {
            // 无法检测到直线，返回图像底部
            return actualRoi.y + roiImage.rows;
        }
        
        // 找到最底部的近似水平线
        int bottomLineY = 0;
        for (const auto& line : lines) {
            int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
            // 检查是否接近水平
            if (std::abs(y1 - y2) < 20) {
                double avgY = (y1 + y2) / 2.0;
                if (avgY > bottomLineY) {
                    bottomLineY = static_cast<int>(avgY);
                }
            }
        }
        
        if (bottomLineY == 0) {
            // 没有找到合适的水平线，返回图像底部
            return actualRoi.y + roiImage.rows;
        }
        
        return bottomLineY + actualRoi.y;
    }
    
    /**
     * 测量关键毁伤参数
     */
    static std::map<std::string, double> measureDamageParameters(
        const std::vector<cv::Point>& contour,
        int groundLine,
        double pixelToMeterRatio,
        int imageHeight) {
        
        // 获取轮廓的基本信息
        cv::Rect boundingRect = cv::boundingRect(contour);
        int x = boundingRect.x, y = boundingRect.y, w = boundingRect.width, h = boundingRect.height;
        
        // 提取轮廓点的x坐标用于后续计算
        std::vector<cv::Point> contourPoints = contour;
        
        // 1. 计算缺口上沿宽度(顶部10%区域)
        double topY = y + h * 0.1;
        std::vector<cv::Point> topPoints;
        for (const auto& pt : contourPoints) {
            if (pt.y <= topY) {
                topPoints.push_back(pt);
            }
        }
        
        double upperWidth = 0;
        if (!topPoints.empty()) {
            int topMinX = topPoints[0].x;
            int topMaxX = topPoints[0].x;
            for (const auto& pt : topPoints) {
                topMinX = std::min(topMinX, pt.x);
                topMaxX = std::max(topMaxX, pt.x);
            }
            upperWidth = (topMaxX - topMinX) * pixelToMeterRatio;
        }
        
        // 2. 计算缺口下沿宽度(底部10%区域，但不低于地面)
        double bottomY = y + h * 0.9;
        std::vector<cv::Point> bottomPoints;
        for (const auto& pt : contourPoints) {
            if (y + h * 0.8 <= pt.y && pt.y <= std::min(bottomY, static_cast<double>(groundLine))) {
                bottomPoints.push_back(pt);
            }
        }
        
        double lowerWidth = 0;
        if (!bottomPoints.empty()) {
            int bottomMinX = bottomPoints[0].x;
            int bottomMaxX = bottomPoints[0].x;
            for (const auto& pt : bottomPoints) {
                bottomMinX = std::min(bottomMinX, pt.x);
                bottomMaxX = std::max(bottomMaxX, pt.x);
            }
            lowerWidth = (bottomMaxX - bottomMinX) * pixelToMeterRatio;
        }
        
        // 3. 计算缺口距地面高度
        int damageBottom = std::min(y + h, groundLine);
        double heightFromGround = std::max(0, groundLine - damageBottom) * pixelToMeterRatio;
        
        // 4. 计算爆堆最大高度
        // 找到毁伤区域内的最高点(相对于地面)
        std::vector<cv::Point> pilePoints;
        for (const auto& pt : contourPoints) {
            if (pt.y < groundLine) {
                pilePoints.push_back(pt);
            }
        }
        
        double pileHeight = 0;
        if (!pilePoints.empty()) {
            int maxPileY = pilePoints[0].y;
            for (const auto& pt : pilePoints) {
                maxPileY = std::min(maxPileY, pt.y);
            }
            pileHeight = (groundLine - maxPileY) * pixelToMeterRatio;
        }
        
        // 5. 计算毁伤面积
        double damageArea = cv::contourArea(contour) * (pixelToMeterRatio * pixelToMeterRatio);
        
        std::map<std::string, double> result;
        result["upper_width"] = std::round(upperWidth * 100) / 100;  // 保留两位小数
        result["lower_width"] = std::round(lowerWidth * 100) / 100;
        result["height_from_ground"] = std::round(heightFromGround * 100) / 100;
        result["pile_height"] = std::round(pileHeight * 100) / 100;
        result["damage_area"] = std::round(damageArea * 100) / 100;
        
        return result;
    }
    
    /**
     * 在帧上可视化毁伤分析结果
     */
    static cv::Mat visualizeDamageAnalysis(const cv::Mat& frame,
                                         const std::vector<cv::Point>& damageContour,
                                         const std::map<std::string, double>& damageParams,
                                         int groundLine) {
        cv::Mat visFrame = frame.clone();
        
        // 绘制地面线
        cv::line(visFrame, cv::Point(0, groundLine), cv::Point(frame.cols, groundLine), 
                cv::Scalar(0, 255, 0), 2);
        
        // 绘制毁伤轮廓
        cv::drawContours(visFrame, std::vector<std::vector<cv::Point>>{damageContour}, -1, 
                        cv::Scalar(0, 0, 255), 2);
        
        // 获取参数用于显示位置
        cv::Rect bbox = cv::boundingRect(damageContour);
        int x = bbox.x, y = bbox.y, w = bbox.width, h = bbox.height;
        
        // 在图像上显示测量结果
        int textY = std::max(50, y - 20);
        cv::putText(visFrame, "上沿宽度: " + std::to_string(damageParams.at("upper_width")) + "m",
                   cv::Point(x, textY), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(visFrame, "下沿宽度: " + std::to_string(damageParams.at("lower_width")) + "m",
                   cv::Point(x, textY + 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(visFrame, "距地高度: " + std::to_string(damageParams.at("height_from_ground")) + "m",
                   cv::Point(x, textY + 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(visFrame, "爆堆高度: " + std::to_string(damageParams.at("pile_height")) + "m",
                   cv::Point(x, textY + 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        return visFrame;
    }
    
    /**
     * 扩展检测框区域
     */
    static cv::Rect expandRoi(int x1, int y1, int x2, int y2, 
                             const cv::Size& frameShape, 
                             double expansionRatio = 0.1) {
        int height = frameShape.height;
        int width = frameShape.width;
        
        // 计算原始宽度和高度
        int w = x2 - x1;
        int h = y2 - y1;
        
        // 计算扩展量
        int expandW = static_cast<int>(w * expansionRatio);
        int expandH = static_cast<int>(h * expansionRatio);
        
        // 计算扩展后的坐标
        int newX1 = std::max(0, x1 - expandW);
        int newY1 = std::max(0, y1 - expandH);
        int newX2 = std::min(width, x2 + expandW);
        int newY2 = std::min(height, y2 + expandH);
        
        return cv::Rect(newX1, newY1, newX2 - newX1, newY2 - newY1);
    }
    
    /**
     * 寻找轮廓内的最大内接矩形
     */
    static cv::Rect findLargestInscribedRectangle(const std::vector<cv::Point>& contour) {
        // 获取轮廓的最小外接矩形
        cv::Rect boundingRect = cv::boundingRect(contour);
        int xMin = boundingRect.x, yMin = boundingRect.y, w = boundingRect.width, h = boundingRect.height;
        
        if (w <= 0 || h <= 0) {
            return cv::Rect(0, 0, 0, 0);
        }
        
        // 创建一个掩码，只包含这个轮廓
        cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
        std::vector<std::vector<cv::Point>> shiftedContour;
        std::vector<cv::Point> shiftedPoints;
        
        for (const auto& pt : contour) {
            shiftedPoints.push_back(cv::Point(pt.x - xMin, pt.y - yMin));
        }
        shiftedContour.push_back(shiftedPoints);
        
        cv::drawContours(mask, shiftedContour, -1, cv::Scalar(255), -1);
        
        // 转换为二值掩码（0和1）
        cv::Mat binaryMask;
        cv::threshold(mask, binaryMask, 127, 1, cv::THRESH_BINARY);
        
        // 如果掩码全黑，直接返回
        int nonZeroCount = cv::countNonZero(binaryMask);
        if (nonZeroCount == 0) {
            return cv::Rect(xMin, yMin, 0, 0);
        }
        
        // 计算每个像素点向上连续为1的最大高度
        std::vector<std::vector<int>> height(h, std::vector<int>(w, 0));
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                if (binaryMask.at<uchar>(i, j) == 1) {
                    height[i][j] = (i == 0) ? 1 : height[i-1][j] + 1;
                }
            }
        }
        
        // 寻找最大矩形
        int maxArea = 0;
        cv::Rect bestRect(0, 0, 0, 0);
        
        for (int i = 0; i < h; ++i) {
            std::vector<int> heights = height[i];
            std::vector<int> stack;
            
            for (int j = 0; j <= w; ++j) {
                int currentHeight = (j < w) ? heights[j] : 0;
                
                while (!stack.empty() && heights[stack.back()] > currentHeight) {
                    int k = stack.back();
                    stack.pop_back();
                    
                    int width = stack.empty() ? j : j - stack.back() - 1;
                    int area = heights[k] * width;
                    
                    if (area > maxArea) {
                        maxArea = area;
                        int rectX = stack.empty() ? 0 : stack.back() + 1;
                        int rectY = i - heights[k] + 1;
                        bestRect = cv::Rect(rectX, rectY, width, heights[k]);
                    }
                }
                stack.push_back(j);
            }
        }
        
        // 转换回原始图像坐标系
        return cv::Rect(xMin + bestRect.x, yMin + bestRect.y, bestRect.width, bestRect.height);
    }
};

/**
 *  毁伤分析主函数
 *
 */
std::map<std::string, cv::Mat> analyzeDamage(const cv::Mat& beforeFrame,
                                           const cv::Mat& afterFrame,
                                           const std::vector<int>& detectionBox,
                                           const std::vector<int>& cropBox,
                                           int threshold = 32) {
    // 验证输入参数
    if (detectionBox.size() != 4 || cropBox.size() != 4) {
        throw std::invalid_argument("Detection box and crop box must have 4 coordinates");
    }
    
    int x1 = detectionBox[0], y1 = detectionBox[1], x2 = detectionBox[2], y2 = detectionBox[3];
    
    // 计算扩展后的ROI（向外扩展1%）
    cv::Rect roi = DamageAnalyzer::expandRoi(x1, y1, x2, y2, beforeFrame.size(), 0.01);

    // 裁剪前后帧的ROI区域
    cv::Rect roiRect(roi.x, roi.y, roi.width, roi.height);
    cv::Mat beforeRoi = beforeFrame(roiRect);
    cv::Mat afterRoi = afterFrame(roiRect);

    // 计算ROI区域的帧差
    cv::Mat diffRoi = DamageAnalyzer::computeFrameDifference(beforeRoi, afterRoi, threshold);

    // 创建墙体掩码（在ROI坐标系中）
    cv::Mat wallMask = cv::Mat::zeros(diffRoi.size(), CV_8UC1);
    int wallX1 = std::max(0, x1 - roi.x);
    int wallY1 = std::max(0, y1 - roi.y);
    int wallX2 = std::min(roi.width, x2 - roi.x);
    int wallY2 = std::min(roi.height, y2 - roi.y);
    cv::rectangle(wallMask, cv::Point(wallX1, wallY1), cv::Point(wallX2, wallY2),
                 cv::Scalar(255), -1);

    // 提取墙体区域内的毁伤轮廓
    int frameArea = diffRoi.rows * diffRoi.cols;
    std::vector<std::vector<cv::Point>> damageContours =
        DamageAnalyzer::extractDamageContours(diffRoi, wallMask, 0.001, frameArea);

    std::map<std::string, cv::Mat> result;
    
    if (!damageContours.empty()) {
        // 创建毁伤区域掩码
        cv::Mat damageMask = cv::Mat::zeros(beforeFrame.size(), CV_8UC3);

        // 将ROI坐标系中的轮廓转换回原图坐标系并填充
        for (const auto& contour : damageContours) {
            std::vector<cv::Point> contourOriginal;
            for (const auto& pt : contour) {
                contourOriginal.push_back(cv::Point(pt.x + roi.x, pt.y + roi.y));  // x坐标加上ROI的x偏移
            }

            // 在掩码上填充毁伤区域（红色）
            cv::drawContours(damageMask, std::vector<std::vector<cv::Point>>{contourOriginal},
                           -1, cv::Scalar(0, 0, 255), -1);
        }
        
        // 创建毁伤区域的二值掩码
        cv::Mat damageMaskGray;
        cv::cvtColor(damageMask, damageMaskGray, cv::COLOR_BGR2GRAY);
        cv::Mat damageBinary;
        cv::threshold(damageMaskGray, damageBinary, 10, 255, cv::THRESH_BINARY);

        // 创建crop_box区域的掩码
        cv::Mat cropMask = cv::Mat::zeros(beforeFrame.size(), CV_8UC1);
        cv::rectangle(cropMask, cv::Point(cropBox[0], cropBox[1]),
                     cv::Point(cropBox[2], cropBox[3]), cv::Scalar(255), -1);

        // 创建复合掩码：只在crop_box内且在毁伤区域内的部分
        cv::Mat combinedMask;
        cv::bitwise_and(damageBinary, cropMask, combinedMask);

        // 生成二值可视化图像
        cv::Mat binaryVisualization = cv::Mat::zeros(beforeFrame.size(), CV_8UC1);
        binaryVisualization.setTo(255, combinedMask > 0);

        // 在二值图像上绘制面积最大的内接矩形
        cv::Mat binaryVisualizationColor;
        cv::cvtColor(binaryVisualization, binaryVisualizationColor, cv::COLOR_GRAY2BGR);

        // 查找面积最大的内接矩形
        cv::Rect largestRect(0, 0, 0, 0);
        int maxArea = 0;

        for (const auto& contour : damageContours) {
            std::vector<cv::Point> contourOriginal;
            for (const auto& pt : contour) {
                contourOriginal.push_back(cv::Point(pt.x + roi.x, pt.y + roi.y));
            }

            // 寻找最大内接矩形
            cv::Rect rect = DamageAnalyzer::findLargestInscribedRectangle(contourOriginal);

            // 计算矩形面积
            int area = rect.width * rect.height;

            // 更新最大面积矩形
            if (area > maxArea) {
                maxArea = area;
                largestRect = rect;
            }
        }
        
        // 如果找到了有效的矩形，绘制它
        if (maxArea > 0) {
            cv::rectangle(binaryVisualizationColor, largestRect, cv::Scalar(0, 255, 0), 2);
        }
        
        result["binary_visualization"] = binaryVisualizationColor;
        result["damage_mask"] = damageMask;
    } else {
        result["binary_visualization"] = cv::Mat::zeros(beforeFrame.size(), CV_8UC1);
        result["damage_mask"] = cv::Mat::zeros(beforeFrame.size(), CV_8UC3);
    }
    
    return result;
}