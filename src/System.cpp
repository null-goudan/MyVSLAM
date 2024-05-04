#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
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
        cout << "Vocabulary loaded!" << endl
             << endl;

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
        mpLocalMapper = new LocalMapping(mpMap);
        mptLocalMapping = new thread(&Goudan_SLAM::LocalMapping::Run, mpLocalMapper);

        // 初始化闭环检测(Loop Closing)线程并启动
        mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, false);
        mptLoopClosing = new thread(&Goudan_SLAM::LoopClosing::Run, mpLoopCloser);

        // 初始化Viewer线程并启动
        // cout << "Init Viewer" <<endl;
        mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile);
        // cout <<"Viewer Start................................"<<endl;
        mptViewer = new thread(&Viewer::Run, mpViewer);
        // cout <<"Viewer Running.............................." <<endl;
        // 设置一些必要的指针给对象
        mpTracker->SetLocalMapper(mpLocalMapper);
        mpTracker->SetLoopClosing(mpLoopCloser);

        mpLocalMapper->SetTracker(mpTracker);
        mpLocalMapper->SetLoopCloser(mpLoopCloser);

        mpLoopCloser->SetTracker(mpTracker);
        mpLoopCloser->SetLocalMapper(mpLocalMapper);
    }

    cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
    {
        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode)
            {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped())
                {
                    // usleep(1000);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                mpTracker->InformOnlyTracking(true); // 定位时，只跟踪
                mbActivateLocalizationMode = false;  // 防止重复执行
            }
            if (mbDeactivateLocalizationMode)
            {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false; // 防止重复执行
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset)
            {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        return mpTracker->GrabImageMonocular(im, timestamp);
    }

    void System::Reset()
    {
        unique_lock<mutex> lock(mMutexReset);
        mbReset = true;
    }

    void System::SaveTrajectoryKITTI(const string &filename)
    {
        cout << endl
             << "Saving camera trajectory to " << filename << " ..." << endl;

        vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<Goudan_SLAM::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++)
        {
            Goudan_SLAM::KeyFrame *pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

            while (pKF->isBad())
            {
                //  cout << "bad parent" << endl;
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw * pKF->GetPose() * Two;

            cv::Mat Tcw = (*lit) * Trw;
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

            f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2) << " " << twc.at<float>(0) << " " << Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " " << twc.at<float>(1) << " " << Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " " << twc.at<float>(2) << endl;
        }
        f.close();
        cout << endl
             << "trajectory saved!" << endl;
    }

}