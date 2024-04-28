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
        mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
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
        mpLocalMapper=pLocalMapper;
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

            // cout << "after initialization Twc:" << endl << mCurrentFrame.mTcw << endl;

            mpFrameDrawer->Update(this);

            if (mState != OK)
                return;
        }
        else
        {
            // bOK为临时变量，用于表示每个函数是否执行成功
            bool bOK;
            // 2. 跟踪 (此时已经初始化完毕)
            // 正常VO模式
            if (!mbOnlyTracking)
            {
                if (mState == OK) // 正常初始化成功
                {
                    // 检查并更新上一帧被替换的MapPoints
                    // :TODO

                    // 2.1 跟踪上一帧或者参考帧或者重定位
                    // 运动模型是空的或刚完成重定位  (重定位未做)  :TODO
                    // 只要mVelocity不为空就选择TrackWithMotionModel
                    if (mVelocity.empty())
                    {
                        // 将上一帧的位姿作为当前帧的初始位姿
                        // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
                        // 优化每个特征点都对应3D点重投影误差即可得到位姿
                        bOK = TrackReferenceKeyFrame();
                        if (bOK)
                        {
                            cout << "Track with no Velocity model successfully!" << endl;
                            cout << "Now Camera Pose:" << endl
                                 << mCurrentFrame.mTcw << endl;
                        }
                    }
                    else
                    {
                        // 根据恒速模型设定当前帧的位姿
                        // 通过投影的方式在参考帧中找当前帧的匹配点
                        // 优化每个特征点所对应的3D点的投影误差即可得到位姿
                        bOK = TrackWithMotionModel();
                        if (!bOK)
                        {
                            bOK = TrackReferenceKeyFrame();
                            if (bOK)
                            {
                                cout << "Track with no Velocity model successfully!" << endl;
                                cout << "Now Camera Pose:" << endl
                                     << mCurrentFrame.mTcw << endl;
                            }
                        }
                        else
                        {
                            cout << "Track with Velocity model successfully!" << endl;
                            cout << "Now Camera Pose:" << endl
                                 << mCurrentFrame.mTcw << endl;
                        }
                    }
                }
                else
                {
                }
            }
            else
            {
                // else 只定位模式 :TODO
            }

            // 将最新的关键帧作为reference frame
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
            // // 2.2 在帧间匹配得到初始的姿态后，进行localmap获得更多的匹配点，得到更加精准的相机位姿
            // // 在2.1中主要是两两跟踪（恒速模型跟踪上一帧、跟踪参考帧），这里搜索局部关键帧后搜集所有局部MapPoints，
            // // 然后将局部MapPoints和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
            // if (!mbOnlyTracking)
            // {
            //     if (bOK)
            //         bOK = TrackLocalMap();
            // }
            // else
            // {
            //     // 重定位
            //     if (bOK && !mbVO)
            //         bOK = TrackLocalMap();
            // }

            // Update drawer
            mpFrameDrawer->Update(this);

            // 如果跟踪是好的, 检查是否要插入新的关键帧
            if (bOK)
            {
                // Update motion model
                if (!mLastFrame.mTcw.empty())
                {
                    // 2.3：更新恒速运动模型TrackWithMotionModel中的mVelocity
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));

                    cout << "Last Twc :" << endl << LastTwc << endl;
                    cout << "mCurrent Frame Tcw :" <<endl <<  mCurrentFrame.mTcw <<endl;
                    mVelocity = mCurrentFrame.mTcw * LastTwc; // Tcl

                    cout << "velocity : " << mVelocity  <<endl;
                }
                else
                    mVelocity = cv::Mat();

                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                // 2.4 清除UpdateLastFrame中当前帧临时添加的MapPoints
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

                // // 2.6：检测并插入关键帧，对于双目会产生新的MapPoints
                // if(NeedNewKeyFrame())
                //     CreateNewKeyFrame();
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // 保存上一帧的数据
            mLastFrame = Frame(mCurrentFrame);
        }

        // 步骤3: 记录位姿信息. 用于轨迹复现
        if (!mCurrentFrame.mTcw.empty())
        {
            // 计算相对姿态T_currentFrame_referenceKeyFrame
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPose();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        }
        else
        {
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
        // Optimizer::GlobalBundleAdjustemnt(mpMap,20);

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

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame); // 重定位用的上一帧信息

        // cout <<"[Initialization info ]: mLastFrame Tcw"  << endl<< mLastFrame.mTcw <<endl;
        // cout << "[Initialization info ]: mCurrentFrame Tcw"  << endl<< mCurrentFrame.mTcw<<endl;


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

        cout << "mCurrentFrame feature num is:" << mCurrentFrame.N << endl;

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
                    // pMP->mbTrackInView = false;
                    // pMP->mnLastFrameSeen = mCurrentFrame.mnId;
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

        UpdateLastFrame();

        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        mpMapDrawer->SetCurrentCameraPose(mVelocity * mLastFrame.mTcw);

        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        int th = 15;

        // 2. 根据匀速模型进行对上一帧的MapPoints进行跟踪
        // 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th);

        // 如果跟踪的点少，则扩大搜索半径再来一次
        if (nmatches < 20)
        {
            cout << "ProjectMatch num too less" << endl;
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th);
        }

        if (nmatches < 20)
        {
            cout << "ProjectMatch num too less" << endl;
            return false;
        }
        // Optimize frame pose with all matches
        // 步骤3：优化位姿，only-pose BA优化
        Optimizer::PoseOptimization(&mCurrentFrame);

        // 步骤4：优化位姿后剔除outlier的mvpMapPoints
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    // MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    // pMP->mbTrackInView = false;
                    // pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        return nmatchesMap >= 10;
    }

    bool Tracking::TrackLocalMap()
    {
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
        // KeyFrame* pRef = mLastFrame.mpReferenceKF;
        cv::Mat Tlr = mlRelativeFramePoses.back();

        mLastFrame.SetPose(Tlr * pRef->GetPose()); // Tlr*Trw = Tlw 1:last r:reference w:world
    }

    // :TODO
    // bool Tracking::NeedNewKeyFrame()
    // {
    //     if(mbOnlyTracking)
    //         return false;
        
    //     if(->isStoped())
    // }

    void Tracking::UpdateLocalMap()
    {
        // 更新局部关键帧和局部MapPoints
        UpdateLocalKeyFrames();
        UpdateLocalPoints();

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
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
        // 1. 遍历当前帧的MapPoints， 记录所有能观测到当前帧MapPoints的关键帧
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
                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), iend = observations.end(); it != iend; it++)
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

        // 2.更新局部关键帧， 添加局部关键帧有三个策略
        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // 策略1. 能观测到当前帧MapPoints的关键帧作为局部关键帧
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
            // 防止重复添加局部关键帧
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }

        // 策略2. 与策略1得到的局部关键帧共视程度很高的座位局部关键帧
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
            // KeyFrame *pParent = pKF->GetParent();
            // if (pParent)
            // {
            //     // mnTrackReferenceForFrame防止重复添加局部关键帧
            //     if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            //     {
            //         mvpLocalKeyFrames.push_back(pParent);
            //         pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            //         break;
            //     }
            // }
        }

        // 策略3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
        if(pKFmax)
        {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    // 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
    void Tracking::SearchLocalPoints(){
        // 1. 遍历当前帧的mvpMapPoints， 标解这些MapPoints不参与之后的搜索
        // 因为当前的mvpMapPoints一定在当前帧的视野中
        for(vector<MapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit!=vend;vit++)
        {
            MapPoint* pMP = *vit;
            if(pMP)
            {
                if(pMP->isBad())
                {
                    *vit = static_cast<MapPoint*>(NULL);
                }
                else
                {
                    // 更新能观测到该点的帧数加1
                    pMP->IncreaseVisible();
                    // 标记被低昂前阵观测到
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    // 标记该点将来不被投影，因为已经匹配过
                    pMP->mbTrackInView  = false;
                }
            }
        }

        int nToMatch = 0;

        // 2. 将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配
        for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;

            // 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
            if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if(pMP->isBad())
                continue;
            
            // Project (this fills MapPoint variables for matching)
            // 步骤2.1：判断LocalMapPoints中的点是否在在视野内
            if(mCurrentFrame.isInFrustum(pMP,0.5))
            {
                // 观测到该点的帧数加1，该MapPoint在某些帧的视野范围内
                pMP->IncreaseVisible();
                // 只有在视野范围内的MapPoints才参与之后的投影匹配
                nToMatch++;
            }
        }

        if(nToMatch>0)
        {
            ORBmatcher matcher(0.8);
            int th = 1;
            if(mCurrentFrame.mnId<mnLastRelocFrameId + 2)
                th = 5;
            
            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    void Tracking::SetViewer(Viewer *pViewer)
    {
        mpViewer = pViewer;
    }
} // namespace Goudan_SLAM
