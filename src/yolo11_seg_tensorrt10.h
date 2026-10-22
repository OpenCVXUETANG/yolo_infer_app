#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"

struct DetectResult {
	int classId;
	float score;
	cv::Rect box;
};

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;

class YOLO11TRTSegment {
public:
	void initConfig(std::string enginefile, float confidence_threshold);
    void detect(cv::Mat& frame, std::vector<DetectResult> &results);
	float sigmoid_function(float a);
	~YOLO11TRTSegment();
private:
	float score_threshold = 0.25;
	int input_w;
	int input_h;
	int output_h;
	int output_w;
	float sx = 160.0f / 640.0f;
	float sy = 160.0f / 640.0f;
	ICudaEngine* engine{ nullptr };
	IExecutionContext* context{ nullptr };
	IRuntime* runtime{ nullptr };
	void* buffers[3] = { NULL, NULL, NULL };
	std::vector<float> prob;
	std::vector<float> mask;
	cudaStream_t stream;
};
