// ==================== include/rknnPool.hpp ====================
#pragma once

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <mutex>
#include <memory>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "rknn_api.h"
#include "postprocess_pre.h"
#include "preprocess2.h"
#include "BYTETracker.h"

#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"

#include "ThreadPool.hpp"

#ifndef INFO_LOG
#define INFO_LOG(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#endif

#ifndef ERROR_LOG
#define ERROR_LOG(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        ERROR_LOG("fopen %s fail!", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (size <= 0) {
        ERROR_LOG("invalid model file size: %d", size);
        fclose(fp);
        return NULL;
    }
    unsigned char *data = (unsigned char *)malloc(size);
    if (!data) {
        fclose(fp);
        return NULL;
    }
    size_t read_bytes = fread(data, 1, size, fp);
    fclose(fp);
    if (read_bytes != (size_t)size) {
        ERROR_LOG("model read incomplete: %zu/%d bytes", read_bytes, size);
        free(data);
        return NULL;
    }
    *model_size = size;
    return data;
}

class rknn_lite
{
public:
    cv::Mat ori_img;
    int instance_id;

    rknn_lite(char *model_name, int n, int id)
    {
        instance_id = id;
        INFO_LOG("Instance %d: Initializing...", instance_id);

        int model_data_size = 0;
        model_data = load_model(model_name, &model_data_size);
        if (!model_data) {
            ERROR_LOG("Instance %d: Failed to load model", instance_id);
            exit(-1);
        }

        ret = rknn_init(&rkModel, model_data, model_data_size, 0, NULL);
        if (ret < 0) {
            ERROR_LOG("Instance %d: rknn_init error ret=%d", instance_id, ret);
            exit(-1);
        }

        rknn_core_mask core_mask;
        if (n == 0) core_mask = RKNN_NPU_CORE_0;
        else if (n == 1) core_mask = RKNN_NPU_CORE_1;
        else core_mask = RKNN_NPU_CORE_2;

        ret = rknn_set_core_mask(rkModel, core_mask);
        if (ret < 0) {
            ERROR_LOG("Instance %d: rknn_set_core_mask error ret=%d", instance_id, ret);
            exit(-1);
        }

        ret = rknn_query(rkModel, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
        if (ret < 0) {
            ERROR_LOG("Instance %d: rknn_query version error ret=%d", instance_id, ret);
            exit(-1);
        }
        printf("Instance %d: SDK %s | Driver %s\n", instance_id, version.api_version, version.drv_version);

        ret = rknn_query(rkModel, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (ret < 0) {
            ERROR_LOG("Instance %d: rknn_query in_out_num error ret=%d", instance_id, ret);
            exit(-1);
        }

        input_attrs = new rknn_tensor_attr[io_num.n_input];
        memset(input_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_input);
        for (int i = 0; i < io_num.n_input; i++) {
            input_attrs[i].index = i;
            ret = rknn_query(rkModel, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                ERROR_LOG("Instance %d: rknn_query input_attr error ret=%d", instance_id, ret);
                exit(-1);
            }
        }

        output_attrs = new rknn_tensor_attr[io_num.n_output];
        memset(output_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_output);
        for (int i = 0; i < io_num.n_output; i++) {
            output_attrs[i].index = i;
            ret = rknn_query(rkModel, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                ERROR_LOG("Instance %d: rknn_query output_attr error ret=%d", instance_id, ret);
                exit(-1);
            }
        }

        if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
            channel = input_attrs[0].dims[1];
            height  = input_attrs[0].dims[2];
            width   = input_attrs[0].dims[3];
        } else {
            height  = input_attrs[0].dims[1];
            width   = input_attrs[0].dims[2];
            channel = input_attrs[0].dims[3];
        }
        printf("Instance %d: Input dims [%d,%d,%d,%d]\n", instance_id,
               input_attrs[0].dims[0], input_attrs[0].dims[1],
               input_attrs[0].dims[2], input_attrs[0].dims[3]);
        printf("Instance %d: Input size %dx%d\n", instance_id, width, height);

        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type  = RKNN_TENSOR_UINT8;
        inputs[0].size  = width * height * channel;
        inputs[0].fmt   = RKNN_TENSOR_NHWC;
        inputs[0].pass_through = 0;

        input_buffer.resize(static_cast<size_t>(width) * height * channel);

        tracker = std::make_unique<BYTETracker>(30, 90);

        INFO_LOG("Instance %d: Initialization complete", instance_id);
    }

    ~rknn_lite()
    {
        if (input_attrs)  delete[] input_attrs;
        if (output_attrs) delete[] output_attrs;
        if (model_data)   free(model_data);
        if (rkModel)      rknn_destroy(rkModel);
    }

    void set_track_interval(int interval)
    {
        if (interval < 1) {
            interval = 1;
        }
        track_interval = interval;
    }

    const std::vector<STrack> &get_last_tracks() const
    {
        return last_tracks;
    }

    int get_frame_index() const
    {
        return frame_index;
    }

    int interf(bool draw_result = true, bool enable_track = true, bool enable_rga = false)
    {
        if (ori_img.empty()) {
            ERROR_LOG("Instance %d: Empty input frame", instance_id);
            return -1;
        }

        int64_t t0 = now_ms();

        cv::Mat img = ori_img; // BGR input from OpenCV capture

        int img_width  = img.cols;
        int img_height = img.rows;

        LETTER_BOX letter_box;
        letter_box.in_height     = img_height;
        letter_box.in_width      = img_width;
        letter_box.channel       = img.channels();
        letter_box.target_width  = width;
        letter_box.target_height = height;
        compute_letter_box(&letter_box);

        cv::Mat processed_img;
        if (enable_rga && !preprocess_rga(img, processed_img, letter_box)) {
            cv::Mat rgb;
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
            if (img_width != width || img_height != height) {
                cv::resize(rgb, processed_img,
                           cv::Size(letter_box.resize_width, letter_box.resize_height));
                cv::copyMakeBorder(processed_img, processed_img,
                    letter_box.h_pad_top,  letter_box.h_pad_bottom,
                    letter_box.w_pad_left, letter_box.w_pad_right,
                    cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            } else {
                processed_img = rgb;
            }
        } else if (!enable_rga) {
            cv::Mat rgb;
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
            if (img_width != width || img_height != height) {
                cv::resize(rgb, processed_img,
                           cv::Size(letter_box.resize_width, letter_box.resize_height));
                cv::copyMakeBorder(processed_img, processed_img,
                    letter_box.h_pad_top,  letter_box.h_pad_bottom,
                    letter_box.w_pad_left, letter_box.w_pad_right,
                    cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            } else {
                processed_img = rgb;
            }
        }

        if (!processed_img.isContinuous()) {
            processed_img = processed_img.clone();
        }

        if (input_buffer.size() != processed_img.total() * processed_img.elemSize()) {
            ERROR_LOG("Instance %d: Input buffer size mismatch", instance_id);
            return -1;
        }

        if (processed_img.data != input_buffer.data()) {
            memcpy(input_buffer.data(), processed_img.data, input_buffer.size());
        }
        inputs[0].buf = input_buffer.data();

        int64_t t1 = now_ms();

        ret = rknn_inputs_set(rkModel, io_num.n_input, inputs);
        if (ret < 0) {
            return -1;
        }

        std::vector<rknn_output> outputs(io_num.n_output);
        memset(outputs.data(), 0, sizeof(rknn_output) * io_num.n_output);
        for (int i = 0; i < io_num.n_output; i++) {
            outputs[i].want_float = 0;
        }

        ret = rknn_run(rkModel, NULL);
        if (ret < 0) {
            return -1;
        }

        ret = rknn_outputs_get(rkModel, io_num.n_output, outputs.data(), NULL);
        if (ret < 0) {
            return -1;
        }

        int64_t t2 = now_ms();

        const float nms_threshold      = NMS_THRESH;
        const float box_conf_threshold = BOX_THRESH;

        detect_result_group_t detect_result_group;
        post_process(rkModel, outputs.data(), &letter_box, box_conf_threshold, nms_threshold,
                     &detect_result_group, width, height, io_num, output_attrs, true, 1);

        // 只取 cameraid==1 的结果
        // 超大框面积过滤（对齐测试说明 TC-UT-16）：面积占比 > 0.9 的检测框视为不合理并删除。
        const double frame_area = static_cast<double>(img_width) * static_cast<double>(img_height);
        std::vector<Object> objs;
        for (int i = 0; i < detect_result_group.count; ++i) {
            detect_result_t *d = &detect_result_group.results[i];
            if (d->cameraid != 1) continue;
            int bw = d->box.right - d->box.left;
            int bh = d->box.bottom - d->box.top;
            if (frame_area > 0.0 &&
                static_cast<double>(bw) * static_cast<double>(bh) > 0.9 * frame_area) {
                continue;  // 删除超大（面积占比>0.9）的不合理检测框
            }
            Object o;
            o.rect  = cv::Rect(d->box.left, d->box.top, bw, bh);
            o.prob  = d->prop;
            o.label = d->class_index;
            objs.push_back(o);
        }

        if (enable_track) {
            bool do_track = (track_interval <= 1) || (frame_index % track_interval == 0);
            if (do_track) {
                last_tracks = tracker->update(objs);
            }
            if (draw_result) {
                for (const auto &t : last_tracks) {
                    int x  = int(t.tlwh[0]);
                    int y  = int(t.tlwh[1]);
                    int w  = int(t.tlwh[2]);
                    int h  = int(t.tlwh[3]);
                    int id    = t.track_id;
                    int label = t.label;
                    float score = t.score;

                    cv::Scalar color = tracker->get_color(id);
                    const char *label_name = get_label_name(label);

                    char id_text[128];
                    snprintf(id_text, sizeof(id_text), "%s ID:%d %.1f%%", label_name, id, score * 100.0f);

                    cv::rectangle(ori_img, cv::Point(x, y), cv::Point(x + w, y + h),
                                  color, 2);
                    cv::putText(ori_img, id_text, cv::Point(x, y - 6),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
                }
            }
        } else if (draw_result) {
            // 仅绘制检测框（无跟踪）
            for (const auto &o : objs) {
                cv::Scalar color = tracker->get_color(o.label);
                const char *label_name = get_label_name(o.label);
                char text[128];
                snprintf(text, sizeof(text), "%s %.1f%%", label_name, o.prob * 100.0f);
                cv::rectangle(ori_img, o.rect, color, 2);
                cv::putText(ori_img, text, cv::Point((int)o.rect.x, (int)o.rect.y - 6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
            }
        }

        int64_t t3 = now_ms();

        rknn_outputs_release(rkModel, io_num.n_output, outputs.data());

        acc_pre_ms += (t1 - t0);
        acc_infer_ms += (t2 - t1);
        acc_post_ms += (t3 - t2);
        ++stat_frames;
        if (stat_frames % 60 == 0) {
            printf("[STATS] pre=%.2fms infer=%.2fms post+draw=%.2fms\n",
                   acc_pre_ms / 60.0, acc_infer_ms / 60.0, acc_post_ms / 60.0);
            acc_pre_ms = 0.0;
            acc_infer_ms = 0.0;
            acc_post_ms = 0.0;
        }

        ++frame_index;
        return 0;
    }

private:
    void *model_data = nullptr;
    rknn_context rkModel = 0;
    rknn_sdk_version version;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs  = nullptr;
    rknn_tensor_attr *output_attrs = nullptr;
    rknn_input inputs[1];
    int ret;
    int width, height, channel;
    std::unique_ptr<BYTETracker> tracker;
    std::vector<uint8_t> input_buffer;
    double acc_pre_ms = 0.0;
    double acc_infer_ms = 0.0;
    double acc_post_ms = 0.0;
    int stat_frames = 0;
    int frame_index = 0;
    int track_interval = 1;
    std::vector<STrack> last_tracks;

    bool preprocess_rga(const cv::Mat &bgr, cv::Mat &dst, const LETTER_BOX &letter_box)
    {
        if (bgr.empty() || bgr.type() != CV_8UC3) {
            return false;
        }
        if (!bgr.isContinuous()) {
            return false;
        }
        if (letter_box.resize_width <= 0 || letter_box.resize_height <= 0) {
            return false;
        }
        if (letter_box.w_pad_left < 0 || letter_box.h_pad_top < 0) {
            return false;
        }
        if (input_buffer.size() != static_cast<size_t>(width) * height * channel) {
            return false;
        }
        if ((reinterpret_cast<uintptr_t>(bgr.data) % 16) != 0) {
            return false;
        }

        dst = cv::Mat(height, width, CV_8UC3, input_buffer.data());
        memset(dst.data, 114, input_buffer.size());

        int roi_x = letter_box.w_pad_left;
        int roi_y = letter_box.h_pad_top;
        if (roi_x + letter_box.resize_width > width ||
            roi_y + letter_box.resize_height > height) {
            return false;
        }

        uint8_t *roi_ptr = dst.data + (roi_y * width + roi_x) * 3;
        if ((reinterpret_cast<uintptr_t>(roi_ptr) % 16) != 0) {
            return false;
        }
        rga_buffer_t src = wrapbuffer_virtualaddr((void *)bgr.data, bgr.cols, bgr.rows, RK_FORMAT_BGR_888);
        rga_buffer_t dst_roi = wrapbuffer_virtualaddr((void *)roi_ptr, letter_box.resize_width,
                                                      letter_box.resize_height, RK_FORMAT_RGB_888);
        im_rect src_rect = {0, 0, bgr.cols, bgr.rows};
        im_rect dst_rect = {0, 0, letter_box.resize_width, letter_box.resize_height};

        int ret = imcheck(src, dst_roi, src_rect, dst_rect);
        if (IM_STATUS_NOERROR == ret) {
            IM_STATUS status = imresize(src, dst_roi);
            if (status == IM_STATUS_SUCCESS) {
                return true;
            }
        }

        // Fallback: RGA color convert then resize (still zero-copy into ROI)
        cv::Mat rgb_tmp(bgr.rows, bgr.cols, CV_8UC3);
        if (!rgb_tmp.isContinuous()) {
            return false;
        }
        rga_buffer_t rgb_buf = wrapbuffer_virtualaddr((void *)rgb_tmp.data, bgr.cols, bgr.rows, RK_FORMAT_RGB_888);
        ret = imcheck(src, rgb_buf, src_rect, src_rect);
        if (IM_STATUS_NOERROR != ret) {
            return false;
        }
        IM_STATUS cvt_status = imcvtcolor(src, rgb_buf, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
        if (cvt_status != IM_STATUS_SUCCESS) {
            return false;
        }

        rga_buffer_t rgb_src = wrapbuffer_virtualaddr((void *)rgb_tmp.data, bgr.cols, bgr.rows, RK_FORMAT_RGB_888);
        ret = imcheck(rgb_src, dst_roi, src_rect, dst_rect);
        if (IM_STATUS_NOERROR != ret) {
            return false;
        }
        IM_STATUS resize_status = imresize(rgb_src, dst_roi);
        if (resize_status != IM_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }

    static int64_t now_ms()
    {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return static_cast<int64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
    }
};


static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, dims=[%d,%d,%d,%d], size=%d\n",
           attr->index, attr->name, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3], attr->size);
}
