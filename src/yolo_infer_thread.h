#ifndef YOLO_INFER_THREAD_H
#define YOLO_INFER_THREAD_H

#include <QThread>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "yolo11_seg_tensorrt10.h"

class YOLOInferThread : public QThread
{
    Q_OBJECT
public:
    explicit YOLOInferThread(std::string onnx_file, std::string image_file, float conf);

protected:
    void run();

private:
    void procee_one(cv::Mat &image);
    std::string image_file;
    std::string onnx_file;
    std::vector<std::string> classNames;
    std::shared_ptr<YOLO11TRTSegment> detector{ nullptr };

signals:
    void sendResult(cv::Mat, std::vector<DetectResult>);
};
#endif // YOLO_INFER_THREAD_H
