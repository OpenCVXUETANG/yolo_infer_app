# include <yolo11_seg_tensorrt10.h>

class Logger : public ILogger
{
	void log(Severity severity, const char* msg)  noexcept
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

YOLO11TRTSegment::~YOLO11TRTSegment() {
	// 同步结束，释放资源
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	if (!buffers[0]) {
		delete[] buffers;
	}
	std::cout << "destory instance done!" << std::endl;
}

void YOLO11TRTSegment::initConfig(std::string enginepath, float conf) {
	std::ifstream file(enginepath, std::ios::binary);
	char* trtModelStream = NULL;
	int size = 0;
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
	}
	this->runtime = createInferRuntime(gLogger);
	assert(this->runtime != nullptr);
	this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
	assert(this->engine != nullptr);
	this->context = engine->createExecutionContext();
	assert(this->context != nullptr);
	delete[] trtModelStream;

	// 获取输入与输出维度
	auto inputDims = this->engine->getTensorShape("images");
	auto out1Dims = this->engine->getTensorShape("output0");
	auto out2Dims = this->engine->getTensorShape("output1");

	// 获取输入维度信息
	this->input_h = inputDims.d[2];
	this->input_w = inputDims.d[3];
	printf("inputH : %d, inputW: %d \n", this->input_h, this->input_w);

	// 获取输出维度信息
	this->output_h = out1Dims.d[1];
	this->output_w = out1Dims.d[2];
	std::cout << "out data format: " << this->output_h << "x" << this->output_w << std::endl;

	int cns = out2Dims.d[1];
	int mw = out2Dims.d[2];
	int mh = out2Dims.d[3];

	// 创建GPU显存输入/输出缓冲区
	cudaMalloc(&buffers[0], this->input_h * input_w * 3 * sizeof(float));
	cudaMalloc(&buffers[1], this->output_h * this->output_w * sizeof(float));
	cudaMalloc(&buffers[2], cns * mw * mh * sizeof(float));

	// 创建临时缓存输出
	this->prob.resize(this->output_h * this->output_w);
	this->mask.resize(cns * mw * mh);

	// 创建cuda流
	cudaStreamCreate(&stream);
}


float YOLO11TRTSegment::sigmoid_function(float a)
{
	float b = 1. / (1. + exp(-a));
	return b;
}

void YOLO11TRTSegment::detect(cv::Mat& frame, std::vector<DetectResult>  &items) {
	int64 start = cv::getTickCount();
	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));

	float x_factor = image.cols / static_cast<float>(this->input_w);
	float y_factor = image.rows / static_cast<float>(this->input_h);

	cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0), true, false);

	cudaMemcpyAsync(buffers[0], blob.ptr<float>(), 3 * this->input_h * this->input_w * sizeof(float), cudaMemcpyHostToDevice, stream);

	// 推理
	context->executeV2(buffers);

	// 显存到内存
	cudaMemcpyAsync(prob.data(), buffers[1], this->output_h * this->output_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(mask.data(), buffers[2], 32 * 25600 * sizeof(float), cudaMemcpyDeviceToHost, stream);

	cv::Mat dout(this->output_h, this->output_w, CV_32F, (float*)prob.data());
	cv::Mat gmask(32, 25600, CV_32F, (float*)mask.data());
	cv::Mat det_output = dout.t(); // 8400x116

	// 后处理, 1x25200x85
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Mat> masks;

	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, this->output_h - 32);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

		// 置信度 0～1之间
		if (score > this->score_threshold)
		{
			cv::Mat mask2 = det_output.row(i).colRange(this->output_h - 32, this->output_h);
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
			masks.push_back(mask2);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::RNG rng(12345);
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	cv::Mat rgb_mask = cv::Mat::zeros(frame.size(), frame.type());
	for (size_t i = 0; i < indexes.size(); i++) {
		int index = indexes[i];
		int idx = classIds[index];
		float conf = confidences[index];
		cv::Rect box = boxes[index];
		int x1 = std::max(0, box.x);
		int y1 = std::max(0, box.y);
		int x2 = std::max(0, box.br().x);
		int y2 = std::max(0, box.br().y);
		cv::Mat m2 = masks[index];
		cv::Mat m = m2 * gmask;
		for (int col = 0; col < m.cols; col++) {
			m.at<float>(0, col) = sigmoid_function(m.at<float>(0, col));
		}
		cv::Mat m1 = m.reshape(1, 160);
		int mx1 = std::max(0, int((x1 * this->sx) / x_factor));
		int mx2 = std::max(0, int((x2 * sx) / x_factor));
		int my1 = std::max(0, int((y1 * sy) / y_factor));
		int my2 = std::max(0, int((y2 * sy) / y_factor));
		cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
		cv::Mat rm, det_mask;
		cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));
		for (int r = 0; r < rm.rows; r++) {
			for (int c = 0; c < rm.cols; c++) {
				float pv = rm.at<float>(r, c);
				if (pv > 0.5) {
					rm.at<float>(r, c) = 1.0;
				}
				else {
					rm.at<float>(r, c) = 0.0;
				}
			}
		}
		rm = rm * rng.uniform(0, 255);
		rm.convertTo(det_mask, CV_8UC1);
		if ((y1 + det_mask.rows) >= frame.rows) {
			y2 = frame.rows - 1;
		}
		if ((x1 + det_mask.cols) >= frame.cols) {
			x2 = frame.cols - 1;
		}
		cv::Mat mask = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
		det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
		add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);
        DetectResult dr;
        dr.box = box;
        dr.score = conf;
        dr.classId = idx;
		cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 1, 8);
        items.push_back(dr);
	}

	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	cv::addWeighted(frame, 0.6, rgb_mask, 0.4, 0, frame);
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}
