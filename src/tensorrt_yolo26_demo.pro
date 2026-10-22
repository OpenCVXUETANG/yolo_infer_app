QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    yolo11_seg_tensorrt10.cpp \
    yolo_infer_thread.cpp

HEADERS += \
    mainwindow.h \
    yolo11_seg_tensorrt10.h \
    yolo_infer_thread.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../opencv-4.10.0/opencv/build/x64/vc16/lib/ -lopencv_world4100
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../opencv-4.10.0/opencv/build/x64/vc16/lib/ -lopencv_world4100d
else:unix: LIBS += -L$$PWD/../../../opencv-4.10.0/opencv/build/x64/vc16/lib/ -lopencv_world4100

INCLUDEPATH += $$PWD/../../../opencv-4.10.0/opencv/build/include
DEPENDPATH += $$PWD/../../../opencv-4.10.0/opencv/build/include

unix|win32: LIBS += -L$$PWD/../../../TensorRT-10.8/lib/ \
    -lnvinfer_10 -lnvinfer_dispatch_10 -lnvinfer_lean_10 \
    -lnvinfer_plugin_10 -lnvinfer_vc_plugin_10 -lnvonnxparser_10

INCLUDEPATH += $$PWD/../../../TensorRT-10.8/include
DEPENDPATH += $$PWD/../../../TensorRT-10.8/include

unix|win32: LIBS += -L'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/' \
    -lcublas -lcublasLt -lcuda -lcudadevrt -lcudart -lcudart_static \
    -lcudnn -lcudnn64_8 -lcudnn_adv_infer -lcudnn_adv_infer64_8 \
    -lcudnn_adv_train -lcudnn_adv_train64_8 -lcudnn_cnn_infer -lcudnn_cnn_infer64_8 \
    -lcudnn_cnn_train -lcudnn_cnn_train64_8 -lcudnn_ops_infer -lcudnn_ops_infer64_8 \
    -lcudnn_ops_train -lcudnn_ops_train64_8 -lcufft -lcufftw -lcufilt -lcurand \
    -lcusolver -lcusolverMg -lcusparse -lnppc -lnppial -lnppicc -lnppidei -lnppif \
    -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps -lnvblas -lnvjpeg -lnvml \
    -lnvptxcompiler_static -lnvrtc-builtins_static -lnvrtc -lnvrtc_static -lOpenCL

INCLUDEPATH += 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include'
DEPENDPATH += 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include'
