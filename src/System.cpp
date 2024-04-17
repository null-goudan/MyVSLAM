#include "System.h"

#include <thread>
#include <iostream>
#include <iomanip>

bool has_suffix(const std::string &str, const std::string &suffix)
{
    std::size_t index = str.find(suffix, str.size() - suffix.size());
    return (index != std::string::npos);
}

namespace Goudan_SLAM
{
    System::System(const string &strSettingsFile)
    {
        cout << endl
             << "Goudan-SLAM(a Monocular SLAM)" << endl
             << "designed by Goudan[Yang Yixiang].Zheng Zhou University of Light Industry." << endl
             << endl;

        // Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if (!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        // 加载 ORB vocabulary
        // :TODO

        // 创建关键帧数据库(KeyFrame Database)，记录关键帧
        // :TODO

        // 创建空地图
        // mpMap = new Map();

        // 创建绘图窗口(Drawers)，一般用于Viewer
        // :TODO

        // 初始化 Tracking 线程
        mpTracker = new Tracking(this, strSettingsFile);

        // 初始化局部优化(局部地图) (Local Mapping) 线程并启动
        // :TODO

        // 初始化闭环检测(Loop Closing)线程并启动
        // :TODO

        // 初始化Viewer线程并启动
        // :TODO

        // 设置一些必要的指针给对象

    }

    cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp){
        {
            // 检查模式变化
            // :TODO
        }

        return mpTracker->GrabImageMonocular(im, timestamp);
    }
}