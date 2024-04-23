#include "System.h"

#include <thread>
#include <iostream>
#include <iomanip>

namespace Goudan_SLAM
{
    System::System(const string &strVocFile, const string &strSettingsFile)
    {
        cout << endl
             << "Goudan-SLAM(a Monocular SLAM)" << endl
             << "designed by Goudan[Yang Yixiang].Zheng Zhou University of Light Industry." << endl
             << endl;

        // cout << "Setting File Path: "<< strSettingsFile<<endl;
        // Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if (!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        // 加载 ORB vocabulary
        cout << endl
             << "Loading ORB Vocabulary(Powered by ORB-SLAM org). This could take while..." << endl;

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = false;
        bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if (!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Failed to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        // 创建关键帧数据库(KeyFrame Database)，记录关键帧
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        // 创建空地图
        mpMap = new Map();

        // 创建绘图窗口(Drawers)，一般用于Viewer
        mpFrameDrawer = new FrameDrawer(mpMap);
        mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

        // 初始化 Tracking 线程
        mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile);

        // 初始化局部优化(局部地图) (Local Mapping) 线程并启动
        // :TODO

        // 初始化闭环检测(Loop Closing)线程并启动
        // :TODO

        // 初始化Viewer线程并启动
        // cout << "Init Viewer" <<endl;
        mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile);
        //cout <<"Viewer Start................................"<<endl;
        mptViewer = new thread(&Viewer::Run, mpViewer);
        // cout <<"Viewer Running.............................." <<endl;
        // 设置一些必要的指针给对象
        mpTracker->SetViewer(mpViewer);
    }

    cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
    {
        {
            // 检查模式变化
            // :TODO
        }

        return mpTracker->GrabImageMonocular(im, timestamp);
    }
}