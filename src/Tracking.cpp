#include "Tracking.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Frame.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Map.h"

#include <iostream>
#include <cmath>
#include <mutex>

using namespace std;

namespace Goudan_SLAM
{
    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase *pKFDB, const std::string &strSettingPath)
        : mState(NO_IMAGES_YET), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
          mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys), mpViewer(NULL),
          mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
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

    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
    {
        mpLocalMapper = pLocalMapper;
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
            mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
        else
            mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

        // 追踪线程 (Tracking)
        Track();

        return mCurrentFrame.mTcw.clone();
    }

    void Tracking::Track()
    {
        // track包含两部分：估计运动、跟踪局部地图

        // mState为tracking的状态机
        // SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
        // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
        if (mState == NO_IMAGES_YET)
        {
            mState = NOT_INITIALIZED;
        }

        // mLastProcessedState存储了Tracking最新的状态，用于FrameDrawer中的绘制
        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        // 步骤1：初始化
        if (mState == NOT_INITIALIZED)
        {
            MonocularInitialization();

            mpFrameDrawer->Update(this);

            if (mState != OK)
                return;
        }
        else // 步骤2：跟踪
        {
            // System is initialized. Track Frame.

            // bOK为临时变量，用于表示每个函数是否执行成功
            bool bOK;

            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            // 在viewer中有个开关menuLocalizationMode，有它控制是否ActivateLocalizationMode，并最终管控mbOnlyTracking
            // mbOnlyTracking等于false表示正常VO模式（有地图更新），mbOnlyTracking等于true表示用户手动选择定位模式
            if (!mbOnlyTracking)
            {
                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only tracking" mode.

                // 正常初始化成功
                if (mState == OK)
                {
                    // Local Mapping might have changed some MapPoints tracked in last frame
                    // 检查并更新上一帧被替换的MapPoints
                    // 更新Fuse函数和SearchAndFuse函数替换的MapPoints
                    CheckReplacedInLastFrame();

                    // 步骤2.1：跟踪上一帧或者参考帧或者重定位

                    // 运动模型是空的或刚完成重定位
                    // mCurrentFrame.mnId<mnLastRelocFrameId+2这个判断不应该有
                    // 应该只要mVelocity不为空，就优先选择TrackWithMotionModel
                    // mnLastRelocFrameId上一次重定位的那一帧
                    if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                    {
                        // 将上一帧的位姿作为当前帧的初始位姿
                        // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
                        // 优化每个特征点都对应3D点重投影误差即可得到位姿
                        bOK = TrackReferenceKeyFrame();
                    }
                    else
                    {
                        // 根据恒速模型设定当前帧的初始位姿
                        // 通过投影的方式在参考帧中找当前帧特征点的匹配点
                        // 优化每个特征点所对应3D点的投影误差即可得到位姿
                        bOK = TrackWithMotionModel();
                        if (!bOK)
                            // TrackReferenceKeyFrame是跟踪参考帧，不能根据固定运动速度模型预测当前帧的位姿态，通过bow加速匹配（SearchByBow）
                            // 最后通过优化得到优化后的位姿
                            bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // BOW搜索，PnP求解位姿
                    //bOK = Relocalization();
                }
            }
            else
            {
            }

            // 将最新的关键帧作为reference frame
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            // 步骤2.2：在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
            // local map:当前帧、当前帧的MapPoints、当前关键帧与其它关键帧共视关系
            // 在步骤2.1中主要是两两跟踪（恒速模型跟踪上一帧、跟踪参考帧），这里搜索局部关键帧后搜集所有局部MapPoints，
            // 然后将局部MapPoints和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
            if (!mbOnlyTracking)
            {
                if (bOK)
                    bOK = TrackLocalMap();
            }
            else
            {
                // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.

                // 重定位成功
                if (bOK && !mbVO)
                    bOK = TrackLocalMap();
            }

            if (bOK)
                mState = OK;
            else
                mState = LOST;

            // Update drawer
            mpFrameDrawer->Update(this);

            // If tracking were good, check if we insert a keyframe
            if (bOK)
            {
                // Update motion model
                if (!mLastFrame.mTcw.empty())
                {
                    // 步骤2.3：更新恒速运动模型TrackWithMotionModel中的mVelocity
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    mVelocity = mCurrentFrame.mTcw * LastTwc; // Tcl
                }
                else
                    mVelocity = cv::Mat();

                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                // Clean VO matches
                // 步骤2.4：清除UpdateLastFrame中为当前帧临时添加的MapPoints
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        // 排除UpdateLastFrame函数中为了跟踪增加的MapPoints
                        if (pMP->Observations() < 1)
                        {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }

                // Delete temporal MapPoints
                // 步骤2.5：清除临时的MapPoints，这些MapPoints在TrackWithMotionModel的UpdateLastFrame函数里生成（仅双目和rgbd）
                // 步骤2.4中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
                // 这里生成的仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
                {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }
                // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
                mlpTemporalPoints.clear();

                // Check if we need to insert a new keyframe
                // 步骤2.6：检测并插入关键帧，对于双目会产生新的MapPoints
                if (NeedNewKeyFrame())
                    CreateNewKeyFrame();

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                // 删除那些在bundle adjustment中检测为outlier的3D map点
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            // 跟踪失败，并且relocation也没有搞定，只能重新Reset
            if (mState == LOST)
            {
                if (mpMap->KeyFramesInMap() <= 5)
                {
                    cout << "Track lost soon after initialisation, reseting..." << endl;
                    mpSystem->Reset();
                    return;
                }
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // 保存上一帧的数据
            mLastFrame = Frame(mCurrentFrame);
        }

        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        // 步骤3：记录位姿信息，用于轨迹复现
        if (!mCurrentFrame.mTcw.empty())
        {
            // 计算相对姿态T_currentFrame_referenceKeyFrame
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        }
        else
        {
            // This can happen if tracking is lost
            // 如果跟踪失败，则相对位姿使用上一次值
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
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
                mLastFrame = Frame(mCurrentFrame);
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
                // cout << "matcher point is less than 100: num " << nmatches << endl;
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
                // cout << "initial finished" << endl;
                // 6.删除那些无法进行三角化的匹配点
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
                {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                    {
                        mvIniMatches[i] = -1;
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
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // :将关键帧的描述子转为Bow
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // 关键帧插入地图
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // 将3D点包装成MapPoint并添加至MapPoint
        for (size_t i = 0; i < mvIniMatches.size(); i++)
        {
            if (mvIniMatches[i] < 0)
                continue;

            // 创建 MapPoint
            cv::Mat worldPos(mvIniP3D[i]);
            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            // 为MapPoint添加属性:
            // 1. 观测到该点的关键帧
            // 2. 该点的描述子
            // 3. 该点的平均观测方向和深度范围
            // 表示该KeyFrame的哪个特征点可以观测到哪个3D点
            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            // a. 观测
            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            // b 从众多观测到该点的特征点中挑选区分度最高的描述子
            pMP->ComputeDistinctiveDescriptors();
            // c 更新该点的平均观测方向以及观测距离的范围
            pMP->UpdateNormalAndDepth();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            // 地图中添加该MapPoint
            mpMap->AddMapPoint(pMP);
        }

        // 更新帧间链接关系
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // BA优化
        // Bundle Adjustment
        cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

        // 步骤5：BA优化
        Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

        // 将MapPoints中值深度归一化到1，并归一化两帧之间变换
        // 单目传感器无法恢复真实的深度，这里将点云中值深度（欧式距离，不是指z）归一化到1
        // 评估关键帧场景深度，q=2表示中值
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
        {
            cout << "Wrong initialization, reseting..." << endl;
            if (medianDepth < 0)
                cout << "reason : medianDepth < 0" << endl;
            else
                cout << "reason : pKFcur->TrackedMapPoints(1) < 100 " << endl;
            // Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        // 根据点云归一化比例缩放平移量
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        // 把3D点的尺度也归一化到1
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
        {
            if (vpAllMapPoints[iMP])
            {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame); // 重定位用的上一帧信息

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;
        cout << "Initial mode Finished!!!!!" << endl;
    }

    bool Tracking::TrackReferenceKeyFrame()
    {
        // 1. 将当前帧的描述子转化为BoW向量
        mCurrentFrame.ComputeBoW();

        ORBmatcher matcher(0.7, true);
        vector<MapPoint *> vpMapPointMatches; // 存放匹配上的3D点（初始化得到的点范围中的）

        // 2. 通过特征点的Bow加快当前帧和参考帧之间的特征点匹配(通过初始化已经有的一定的3D点,得到这些3D点对应的下一帧的对应的特征点)
        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

        if (nmatches < 15)
        {
            cout << "nmatches less than 15" << endl;
            return false;
        }

        // 将上一帧的位姿作为当前帧位姿的初始值
        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.SetPose(mLastFrame.mTcw);

        // 通过优化3D-2D的重投影误差来获得位姿
        Optimizer::PoseOptimization(&mCurrentFrame);

        // 剔除优化后的outliner匹配点
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            // cout << mCurrentFrame.mvpMapPoints[i] << endl;
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }
        return nmatchesMap >= 0;
    }

    bool Tracking::TrackWithMotionModel()
    {
        ORBmatcher matcher(0.9, true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points
        // 步骤1：对于双目或rgbd摄像头，根据深度值为上一关键帧生成新的MapPoints
        // （跟踪过程中需要将当前帧与上一帧进行特征点匹配，将上一帧的MapPoints投影到当前帧可以缩小匹配范围）
        // 在跟踪过程中，去除outlier的MapPoint，如果不及时增加MapPoint会逐渐减少
        // 这个函数的功能就是补充增加RGBD和双目相机上一帧的MapPoints数
        UpdateLastFrame();

        // 根据Const Velocity Model(认为这两帧之间的相对运动和之前两帧间相对运动相同)估计当前帧的位姿
        // mVelocity为最近一次前后帧位姿之差
        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        // Project points seen in previous frame
        int th = 15;

        // 步骤2：根据匀速度模型进行对上一帧的MapPoints进行跟踪
        // 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th);

        // If few matches, uses a wider window search
        // 如果跟踪的点少，则扩大搜索半径再来一次
        if (nmatches < 20)
        {
            cout << "less than 20 nmatches TrackWithMotionModel!" << endl;
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th); // 2*th
        }

        if (nmatches < 20)
            return false;

        // Optimize frame pose with all matches
        // 步骤3：优化位姿，only-pose BA优化
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        // 步骤4：优化位姿后剔除outlier的mvpMapPoints
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        if (mbOnlyTracking)
        {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        return nmatchesMap >= 10;
    }

    bool Tracking::TrackLocalMap()
    {
        cout << "TrackLocalMap()" << endl;
        // 1. 更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints
        UpdateLocalMap();

        // 2. 在局部地图中查找与当前帧匹配的MapPoints
        SearchLocalPoints();

        // 3. 更新局部所有MapPoint后对位姿再次优化
        Optimizer::PoseOptimization(&mCurrentFrame);
        mnMatchesInliers = 0;

        // 3. 更新当前帧的MapPoints被观测的程度, 并统计跟踪局部地图的效果
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                // 由于当前帧的MapPoints可以被当前帧观测到，其被观测统计量加1
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking)
                    {
                        // 该MapPoint被其它关键帧观测到过
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    }
                    else
                        // 记录当前帧跟踪到的MapPoints，用于统计跟踪效果
                        mnMatchesInliers++;
                }
            }
        }

        // 步骤4：决定是否跟踪成功
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
            return false;

        if (mnMatchesInliers < 30)
            return false;
        else
            return true;
    }

    void Tracking::UpdateLastFrame()
    {
        // 1. 更新最近一帧的位姿
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();

        mLastFrame.SetPose(Tlr * pRef->GetPose()); // Tlr*Trw = Tlw 1:last r:reference w:world
    }

    /**
     * @brief 断当前帧是否为关键帧
     * @return true if needed
     */
    bool Tracking::NeedNewKeyFrame()
    {
        if (mbOnlyTracking)
            return false;

        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;

        const int nKFs = mpMap->KeyFramesInMap();

        // 2. 判断是否距离上一次插入关键帧的时间太短
        // 如果关键帧比较少，则考虑插入关键帧
        // 或距离上一次重定位超过1s，则考虑插入关键帧
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

        // 步骤3：得到参考关键帧跟踪到的MapPoints数量
        // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // 步骤4：查询局部地图管理器是否繁忙
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

        int nMap = 1;
        int nTotal = 1;

        const float ratioMap = (float)nMap / (float)(std::max(1, nTotal));

        // 步骤6：决策是否需要插入关键帧
        // Thresholds
        // 设定inlier阈值，和之前帧特征点匹配的inlier比例
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f; // 关键帧只有一帧，那么插入关键帧的阈值设置很低

        thRefRatio = 0.9f;

        // MapPoints中和地图关联的比例阈值
        float thMapRatio = 0.35f;
        if (mnMatchesInliers > 300)
            thMapRatio = 0.20f;

        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        // 很长时间没有插入关键帧
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        // localMapper处于空闲状态
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        // 阈值比c1c要高，与之前参考帧（最近的一个关键帧）重复度不是太高
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || ratioMap < thMapRatio) && mnMatchesInliers > 15);

        if ((c1a || c1b) && c2)
        {
            if (bLocalMappingIdle)
            {
                return true;
            }
            else
            {
                mpLocalMapper->InterruptBA();
                return false;
            }
        }
        else
            return false;
    }

    /**
     * @brief 创建新的关键帧
     *
     */
    void Tracking::CreateNewKeyFrame()
    {
        if (!mpLocalMapper->SetNotStop(true))
            return;

        // 步骤1：将当前帧构造成关键帧
        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // 步骤2：将当前关键帧设置为当前帧的参考关键帧
        // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);

        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;

        cout << "Create KeyFrame : id " << pKF->mnId << endl;
    }

    void Tracking::UpdateLocalMap()
    {
        // 更新局部关键帧和局部MapPoints
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

    void Tracking::UpdateLocalPoints()
    {
        // 1. 清空局部MapPoints
        mvpLocalMapPoints.clear();

        // 2. 遍历局部关键帧
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
        {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            // 2. 将局部关键帧的MapPoints添加到mvpLocalMapPoints
            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
            {
                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
                // mnTrackReferenceForFrame防止重复添加局部MapPoint
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad())
                {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    void Tracking::UpdateLocalKeyFrames()
    {
        // Each map point vote for the keyframes in which it has been observed
        // 步骤1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧
        map<KeyFrame *, int> keyframeCounter;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP->isBad())
                {
                    // 能观测到当前帧MapPoints的关键帧
                    const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }

        if (keyframeCounter.empty())
            return;

        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        // 步骤2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
        // 先清空局部关键帧
        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        // V-D K1: shares the map points with current frame
        // 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
        {
            KeyFrame *pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max)
            {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(it->first);
            // mnTrackReferenceForFrame防止重复添加局部关键帧
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }

        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        // V-D K2: neighbors to K1 in the covisibility graph
        // 策略2：与策略1得到的局部关键帧共视程度很高的关键帧作为局部关键帧
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
        {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80)
                break;

            KeyFrame *pKF = *itKF;

            // 策略2.1:最佳共视的10帧
            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
            {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad())
                {
                    // mnTrackReferenceForFrame防止重复添加局部关键帧
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            // 策略2.2:自己的子关键帧
            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
            {
                KeyFrame *pChildKF = *sit;
                if (!pChildKF->isBad())
                {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            // 策略2.3:自己的父关键帧
            KeyFrame *pParent = pKF->GetParent();
            if (pParent)
            {
                // mnTrackReferenceForFrame防止重复添加局部关键帧
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // V-D Kref： shares the most map points with current frame
        // 步骤3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
        if (pKFmax)
        {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    // 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
    void Tracking::SearchLocalPoints()
    {
        // 步骤1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
        // 因为当前的mvpMapPoints一定在当前帧的视野中
        for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
            {
                if (pMP->isBad())
                {
                    *vit = static_cast<MapPoint *>(NULL);
                }
                else
                {
                    // 更新能观测到该点的帧数加1
                    pMP->IncreaseVisible();
                    // 标记该点被当前帧观测到
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    // 标记该点将来不被投影，因为已经匹配过
                    pMP->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        // Project points in frame and check its visibility
        // 步骤2：将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;

            // 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->isBad())
                continue;

            // Project (this fills MapPoint variables for matching)
            // 步骤2.1：判断LocalMapPoints中的点是否在在视野内
            if (mCurrentFrame.isInFrustum(pMP, 0.5))
            {
                // 观测到该点的帧数加1，该MapPoint在某些帧的视野范围内
                pMP->IncreaseVisible();
                // 只有在视野范围内的MapPoints才参与之后的投影匹配
                nToMatch++;
            }
        }

        if (nToMatch > 0)
        {
            ORBmatcher matcher(0.8);
            int th = 1;

            // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;

            // 步骤2.2：对视野范围内的MapPoints通过投影进行特征点匹配
            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    /**
     * @brief 检查上一帧中的MapPoints是否被替换
     * keyframe在local_mapping和loopclosure中存在fuse mappoint。
     * 由于这些mappoint被改变了，且只更新了关键帧的mappoint，对于mLastFrame普通帧，也要检查并更新mappoint
     * @see LocalMapping::SearchInNeighbors()
     */
    void Tracking::CheckReplacedInLastFrame()
    {
        for (int i = 0; i < mLastFrame.N; i++)
        {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];

            if (pMP)
            {
                MapPoint *pRep = pMP->GetReplaced();
                if (pRep)
                {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }
    }

    void Tracking::Reset()
{
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    cout << "System Reseting" << endl;

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // // Reset Loop Closing
    // cout << "Reseting Loop Closing...";
    // mpLoopClosing->RequestReset();
    // cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

    void Tracking::SetViewer(Viewer *pViewer)
    {
        mpViewer = pViewer;
    }

    void Tracking::InformOnlyTracking(const bool &flag)
    {
        mbOnlyTracking = flag;
    }

} // namespace Goudan_SLAM
