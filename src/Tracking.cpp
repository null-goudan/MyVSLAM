#include "Tracking.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Frame.h"
#include "ORBmatcher.h"
#include "Map.h"

#include <iostream>
#include <cmath>
#include <mutex>

using namespace std;

namespace Goudan_SLAM
{
        Tracking::Tracking(System *pSys, ORBVocabulary* pVoc , FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,   KeyFrameDatabase *pKFDB,const std::string &strSettingPath)
        :mState(NO_IMAGES_YET), mpInitializer(static_cast<Initializer*>(NULL)), mpORBVocabulary(pVoc), mpViewer(NULL),
        mpKeyFrameDB(pKFDB), mpSystem(pSys),
        mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        //     |fx  0   cx|
        // K = |0   fy  cy|
        //     |0   0   1 |
        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        // 图像矫正系数
        // [k1 k2 p1 p2 k3]
        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

        mMinFrames = 0;
        mMaxFrames = fps;

        cout << endl
             << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;

        // 1:RGB 0:BGR
        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;
        // Load ORB parameters

        // 每一帧提取的特征点数 1000
        int nFeatures = fSettings["ORBextractor.nFeatures"];
        // 图像建立金字塔时的变化尺度 1.2
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        // 尺度金字塔的层数 8
        int nLevels = fSettings["ORBextractor.nLevels"];
        // 提取fast特征点的默认阈值 20
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        // tracking过程都会用到mpORBextractorLeft作为特征点提取器
        mpORBextractorLeft = new ORBExtractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
        mpIniORBextractor = new ORBExtractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl
             << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
    }

    cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
    {
        mImGray = im;
        // 如果是RGB 或者 RGBA 转化成 灰度图像
        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        // 构造Frame
        if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor,  mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        else
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

        // 追踪线程 (Tracking)
        Track();

        return mCurrentFrame.mTcw.clone();
    }

    void Tracking::Track()
    {
        // mState 为 Tracking的状态机

        // 如果图像复位过， 或者第一次运行， 设置为 NO_IMAGE_YET状态
        if (mState == NO_IMAGES_YET)
            mState = NOT_INITIALIZED;

        // mLastProcessedState存储了Tracking最新的状态，用于FrameDrawer中的绘制
        mLastProcessedState = mState;

        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        // 1. 初始化
        if (mState == NOT_INITIALIZED)
        {
            // cout << "Initialize ready." << endl;
            MonocularInitialization();

            mpFrameDrawer->Update(this);

            if (mState != OK)
                return;
        }
        else
        {
            // 2. 跟踪 (此时已经初始化完毕)
            // bool bOK;
            // 正常VO模式
            // if(!mbOnlyTracking)
            if (mState == OK) // 正常初始化成功
            {
                
            }
            else
            {
            }
            // 只定位模式 :TODO
            // else
        }
    }

    void Tracking::MonocularInitialization()
    {
        if (!mpInitializer)
        {
            // 单目初始化帧提取的特征点数必须大于100，否则放弃此帧
            if (mCurrentFrame.mvKeys.size() > 100)
            {
                // 1. 得到用于初始化的第一帧, 初始化需要两帧
                mInitialFrame = Frame(mCurrentFrame);
                // 记录最近的一帧
                mLastFrames = Frame(mCurrentFrame);
                // mvbPrevMatched最大的情况就是所有的特征点都被追踪上
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                {
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;
                }

                // 由当前帧构造initializer sigma:1.0 iterations:200
                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }
        }
        else
        {
            // 尝试初始化
            // 2. 如果当前帧特征点数大于100，则得到用于初始化的第二帧
            // 如果当前帧特征点太少，重新构造初始器
            // 因此只有连续两帧的特征点个数都大于100时, 才能继续进行初始化的过程

            // cout<<"try Initialize"<<endl;
            if ((int)mCurrentFrame.mvKeys.size() <= 100)
            {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }

            // 3. 在mInitialFrame 和 mCurrentFrame 找匹配的特征点对
            // mvbPrevMathed 为前一帧的特征点， 存储了mInitialFrame 中的那些点将进行接下来的匹配
            // mvIniMatches 存储 mInitialFrame, mCurrentFrame之间匹配的特征点
            // cout<<"Initialize matching.."<<endl;
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);
            // cout << "inital matcher: num " << nmatches <<endl;

            // 4.初始化两帧之间的匹配点太少，重新初始化
            if (nmatches < 100)
            {
                cout << "matcher point is less than 100: num " << nmatches <<endl;
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                return;
            }

            cv::Mat Rcw;                 // 现在的相机的旋转矩阵 相对于世界的
            cv::Mat tcw;                 // 现在相机的平移
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

            // 5. 通过F模型或者H模型进行单目初始化,得到两帧间相对运动、初始MapPoints
            if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
            {
                cout<<"initial finished"<<endl;
                // 6.删除那些无法进行三角化的匹配点
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
                {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                    {
                        mvIniMatches[i] = 1;
                        nmatches--;
                    }
                }
                // 将初始化的第一帧作为世界坐标系, 因此第一帧变换矩阵为单位矩阵
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                // 有Rcw和tcw构造Tcw，并赋值给mTcw, mTcw为世界坐标系到该帧的变换矩阵
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);
                
                // 将三角化得到的3D点包装秤MapPoints
                // Initialize函数会得到mvIniP3D
                CreateInitialMapMonocular();
            }
        }
    }
    
    void Tracking::CreateInitialMapMonocular()
    {
        KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // :将关键帧的描述子转为Bow
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // 关键帧插入地图
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // 将3D点包装秤MapPoint
        for(size_t i = 0;i< mvIniMatches.size(); i++){

        }
    }

    void Tracking::SetViewer(Viewer* pViewer){
        mpViewer=pViewer;
    }
}// namespace Goudan_SLAM
