#include <yolo_infer_thread.h>
#include <QFileInfo>

#pragma execution_character_set("utf-8")

YOLOInferThread::YOLOInferThread(std::string onnx_file, std::string image_file, float conf)
{
    this->onnx_file = onnx_file;
    this->image_file = image_file;
    detector = std::make_shared<YOLO11TRTSegment>();
    detector->initConfig(onnx_file, 0.25);

    std::string label_map = "D:/python/yolov5-7.0/classes.txt";
        std::ifstream fp(label_map);
        std::string name;
        while (!fp.eof()) {
            getline(fp, name);
            if (name.length()) {
                classNames.push_back(name);
            }
        }
        fp.close();
}

void YOLOInferThread::run()
{
    std::cout<<"started yolo inference thread  successfully..."<<std::endl;
    cv::Mat image = cv::imread(this->image_file);
    procee_one(image);
    std::vector<DetectResult> results;
    emit this->sendResult(cv::Mat(), results);
    std::cout<<"end yolo inference thread !!! "<<std::endl;
}

void YOLOInferThread::procee_one(cv::Mat &frame) {
    std::vector<DetectResult> results;
    detector->detect(frame, results);
    for (DetectResult dr : results) {
        cv::Rect box = dr.box;
        cv::putText(frame, cv::format("%s %.2f", classNames[dr.classId].c_str(), dr.score), cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 255, 0));
    }
    emit this->sendResult(frame, results);
}
