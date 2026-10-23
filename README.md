### YOLO Infer APP
QT5 + TensorRT10.x + OpenCV4.10 + VS2022
Jetson Orin Nano C++ supported!
YOLO Segmenat Models including YOLOv8, YOLO11, YOLO26 can deploy with it

### 第三方库依赖
```
- OpenCV4.x\OpenCV5.x
- CUDA11.x\CUDA12.x\CUDA13.x
- TensorRT8.x\TensorRT10.x
```
![](ui_form.jpg)
![](run_demo.jpg)

#### 安装指南
下载代码，导入到新建的QT Creator项目中即可，选择全部覆盖，然后自己依赖库安装与路径修改即可运行。

### 模型转换
模型提供了ONNX格式，engine格式模型跟设备算力是强相关，无法做到设备兼容，所以需要使用下面的命令行转换为engine格式：
trtexec.exe --onnx=yolov8_cu.onnx --saveEngine=yolov8_cu.engine

### jetson orin nano移植指南(TensorRT8.5.x版本代码修改）
把yolo11_seg_tensorrt10.cpp 代码中的从44行 到 66行替换为下面的代码
```cpp
int input_index = engine->getBindingIndex("images");
int output_index = engine->getBindingIndex("output0");
int mask_index = engine->getBindingIndex("output1");

// 获取输入维度信息 NCHW=1x3x640x640
this->input_h = engine->getBindingDimensions(input_index).d[2];
this->input_w = engine->getBindingDimensions(input_index).d[3];
printf("inputH : %d, inputW: %d \n", input_h, input_w);

// 获取输出维度信息
this->output_h = engine->getBindingDimensions(output_index).d[1];
this->output_w = engine->getBindingDimensions(output_index).d[2];
std::cout << "out data format: " << output_h << "x" << output_w << std::endl;

// 创建GPU显存输入/输出缓冲区
std::cout << " input/outpu : " << engine->getNbBindings() << std::endl;
cudaMalloc(&buffers[input_index], this->input_h * this->input_w * 3 * sizeof(float));
cudaMalloc(&buffers[output_index], this->output_h *this->output_w * sizeof(float));
cudaMalloc(&buffers[mask_index], 32 * 25600 * sizeof(float));
```
推理函数从
```cpp
context->executeV2(buffers);
```
替换为
```cpp
// 推理
context->enqueueV2(buffers, stream, nullptr);
```
其它保持不变！

### OpenCV学堂课程主页
https://appetjpz4tj8166.h5.xet.pomoho.com/

### OpenCV实验大师工具软件下载地址（OpenCV算子工作流设计软件）- 腾讯网盘
https://share.weiyun.com/tgoxerG9

### YOLO、OpenCV C++教程 - QT集成版本
https://appetjpz4tj8166.h5.xet.pomoho.com/v1/goods/goods_detail/p_66d65341e4b023c06c898401?type=3&channel_id=

### 合作联系
OpenCV与深度学习开发合作

微信: bfnh2qxx (注明：合作)

技术讨论QQ群: 934105010 (注明: 学习)


