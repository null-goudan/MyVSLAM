#include "MapPoint.h"

#include <mutex>
#include <map>
#include "ORBmatcher.h"

namespace Goudan_SLAM
{
    long unsigned int MapPoint::nNextId = 0;
    mutex MapPoint::mGlobalMutex;

    // 给定坐标和关键帧构造地图点
    MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap)
        : mnFirstKFid(pRefKF->mnID), mnFirstFrame(pRefKF->mnFrameId), mpMap(pMap), mpRefKF(pRefKF), mbBad(false)
    {
        Pos.copyTo(mWorldPos);
        mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    void MapPoint::SetWorldPos(const cv::Mat &Pos)
    {
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        Pos.copyTo(mWorldPos);
    }

    cv::Mat MapPoint::GetWorldPos()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    cv::Mat MapPoint::GetNormal()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mNormalVector.clone();
    }

    KeyFrame *MapPoint::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    void MapPoint::AddObservation(KeyFrame *pKF, size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return;

        mObservations[pKF] = idx;
        nObs++;
    }

    int MapPoint::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    void MapPoint::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF))
            {
                int idx = mObservations[pKF];
                nObs--;
                mObservations.erase(pKF);
                // 如果该keyFrame是参考帧，该Frame被删除后重新制定RefFrame
                if (mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;

                // 当观测到该点的相机数目少于2时，丢弃该点
                if (nObs <= 2)
                    bBad = true;
            }
        }
        if (bBad)
            SetBadFlag();
    }

    map<KeyFrame *, size_t> MapPoint::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    // 告知可以观测到该MapPoint的Frame，该点已被删除
    void MapPoint::SetBadFlag()
    {
        map<KeyFrame *, size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad = true;
            obs = mObservations;   // 将mObservations 转存到 obs
            mObservations.clear(); // 把mObservations 指向的内存清理
        }
        for (map<KeyFrame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            pKF->EraseMapPointMatch(mit->second);
        }

        mpMap->EraseMapPoint(this);
    }

    // 没有经过MapPointCulling检测的MapPoints
    bool MapPoint::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }

    void KeyFrame::EraseMapPointMatch(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
    }

    void KeyFrame::EraseMapPointMatch(MapPoint *pMP)
    {
        int idx = pMP->GetIndexInKeyFrame(this);
        if (idx >= 0)
            mvpMapPoints[idx] = static_cast<MapPoint *>(NULL);
    }

    int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    // 更新平均观测方向以及观测距离范围
    void MapPoint::UpdateNormalAndDepth()
    {

        map<KeyFrame *, size_t> observations;
        KeyFrame *pRefKF;
        cv::Mat Pos;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            if (mbBad)
                return;

            observations = mObservations; // 获得观测到该3d点的所有关键帧
            pRefKF = mpRefKF;             // 观测到该点的参考关键帧
            Pos = mWorldPos.clone();      // 3d点在世界坐标系中的位置
        }

        if (observations.empty())
            return;

        cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
        int n = 0;
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            cv::Mat Owi = pKF->GetCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            normal = normal + normali / cv::norm(normali); // 对所有关键帧对该点的观测方向归一化为单位向量进行求和
            n++;
        }

        cv::Mat PC = Pos - pRefKF->GetCameraCenter(); // 参考关键帧相机指向3D点的向量（在世界坐标系下的表示）
        const float dist = cv::norm(PC);              // 该点到参考关键帧相机的距离
        const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
        const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        const int nLevels = pRefKF->mnScaleLevels; // 金字塔层数

        {
            unique_lock<mutex> lock3(mMutexPos);
            // 另见PredictScale函数前的注释
            mfMaxDistance = dist * levelScaleFactor;                             // 观测到该点的距离最大值
            mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1]; // 观测到该点的距离最小值
            mNormalVector = normal / n;                                          // 获得平均的观测方向
        }
    }

    /**
     * @brief 计算具有代表的描述子
     *
     * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要判断是否更新当前点的最适合的描述子 \n
     * 先获得当前点的所有描述子，然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值
     *
     */
    void MapPoint::ComputeDistinctiveDescriptors()
    {
        // Retrieve all observed descriptors
        vector<cv::Mat> vDescriptors;

        map<KeyFrame *, size_t> observations;

        {
            unique_lock<mutex> lock1(mMutexFeatures);
            if (mbBad)
                return;
            observations = mObservations;
        }

        if (observations.empty())
            return;

        vDescriptors.reserve(observations.size());

        // 遍历观测到3d点的所有关键帧，获得orb描述子，并插入到vDescriptors中
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;

            if (!pKF->isBad())
                vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
        }

        if (vDescriptors.empty())
            return;

        // Compute distances between them
        // 获得这些描述子两两之间的距离
        const size_t N = vDescriptors.size();

        // float Distances[N][N];
        std::vector<std::vector<float>> Distances;
        Distances.resize(N, vector<float>(N, 0));
        for (size_t i = 0; i < N; i++)
        {
            Distances[i][i] = 0;
            for (size_t j = i + 1; j < N; j++)
            {
                int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for (size_t i = 0; i < N; i++)
        {
            // 第i个描述子到其它所有所有描述子之间的距离
            // vector<int> vDists(Distances[i],Distances[i]+N);
            vector<int> vDists(Distances[i].begin(), Distances[i].end());
            sort(vDists.begin(), vDists.end());

            // 获得中值
            int median = vDists[0.5 * (N - 1)];

            // 寻找最小的中值
            if (median < BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }

        {
            unique_lock<mutex> lock(mMutexFeatures);

            // 最好的描述子，该描述子相对于其他描述子有最小的距离中值
            // 简化来讲，中值代表了这个描述子到其它描述子的平均距离
            // 最好的描述子就是和其它描述子的平均距离最小
            mDescriptor = vDescriptors[BestIdx].clone();
        }
    }

    cv::Mat MapPoint::GetDescriptor()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mDescriptor.clone();
    }

    /**
     * @brief Increase Visible
     *
     * Visible表示：
     * 1. 该MapPoint在某些帧的视野范围内，通过Frame::isInFrustum()函数判断
     * 2. 该MapPoint被这些帧观测到，但并不一定能和这些帧的特征点匹配上
     *    例如：有一个MapPoint（记为M），在某一帧F的视野范围内，
     *    但并不表明该点M可以和F这一帧的某个特征点能匹配上
     */
    void MapPoint::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    /**
     * @brief Increase Found
     *
     * 能找到该点的帧数+n，n默认为1
     * @see Tracking::TrackLocalMap()
     */
    void MapPoint::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float MapPoint::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }

    float MapPoint::GetMinDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 0.8f * mfMinDistance;
    }

    float MapPoint::GetMaxDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 1.2f * mfMaxDistance;
    }

    //              ____
    // Nearer      /____\     level:n-1 --> dmin
    //            /______\                       d/dmin = 1.2^(n-1-m)
    //           /________\   level:m   --> d
    //          /__________\                     dmax/d = 1.2^m
    // Farther /____________\ level:0   --> dmax
    //
    //           log(dmax/d)
    // m = ceil(------------)
    //            log(1.2)
    int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF)
    {
        float ratio;
        {
            unique_lock<mutex> lock(mMutexPos);
            // mfMaxDistance = ref_dist*levelScaleFactor为参考帧考虑上尺度后的距离
            // ratio = mfMaxDistance/currentDist = ref_dist/cur_dist
            ratio = mfMaxDistance / currentDist;
        }

        // 同时取log线性化
        int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pKF->mnScaleLevels)
            nScale = pKF->mnScaleLevels - 1;

        return nScale;
    }

    int MapPoint::PredictScale(const float &currentDist, Frame *pF)
    {
        float ratio;
        {
            unique_lock<mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pF->mnScaleLevels)
            nScale = pF->mnScaleLevels - 1;

        return nScale;
    }

}