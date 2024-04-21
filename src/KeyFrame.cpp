#include "KeyFrame.h"
#include "Converter.h"

#include "ORBmatcher.h"
#include <mutex>

namespace Goudan_SLAM
{
    long unsigned int KeyFrame::nNextId = 0;
    KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB)
        : mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
          mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
          mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
          mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
          fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
          mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mDescriptors(F.mDescriptors.clone()),
          mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
          mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
          mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
          mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
          mpORBvocabulary(F.mpORBvocabulary), mbNotErase(false),
          mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb / 2), mpMap(pMap)
    {
        mnID = nNextId;

        mGrid.resize(mnGridCols);
        for (int i = 0; i < mnGridCols; i++)
        {
            mGrid[i].resize(mnGridRows);
            for (int j = 0; j < mnGridRows; j++)
                mGrid[i][j] = F.mGrid[i][j];
        }

        SetPose(F.mTcw);
    }

    void KeyFrame::SetPose(const cv::Mat &Tcw_)
    {
        unique_lock<mutex> lock(mMutexPose);
        Tcw_.copyTo(Tcw);
        cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
        cv::Mat Rwc = Rcw.t();
        Ow = -Rwc * tcw;

        Twc = cv::Mat::eye(4, 4, Tcw.type());
        Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
        Ow.copyTo(Twc.rowRange(0, 3).col(3));
        cv::Mat center = (cv::Mat_<float>(4, 1) << mHalfBaseline, 0, 0, 1);
        Cw = Twc * center;
    }

    cv::Mat KeyFrame::GetPose()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.clone();
    }

    cv::Mat KeyFrame::GetPoseInverse()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Twc.clone();
    }

    cv::Mat KeyFrame::GetCameraCenter()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Ow.clone();
    }

    cv::Mat KeyFrame::GetRotation()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.rowRange(0, 3).colRange(0, 3).clone();
    }

    cv::Mat KeyFrame::GetTranslation()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.rowRange(0, 3).col(3).clone();
    }

    // 在关键帧中添加MapPoints
    void KeyFrame::AddMapPoint(MapPoint *pMp, const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = pMp;
    }

    void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP)
    {
        mvpMapPoints[idx] = pMP;
    }

    void KeyFrame::ComputeBoW()
    {
        if (mBowVec.empty() || mFeatVec.empty())
        {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            // Feature vector associate features with nodes in the 4th level (from leaves up)
            // We assume the vocabulary tree has 6 levels, change the 4 otherwise
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void KeyFrame::SetBadFlag()
    {
        {
            unique_lock<mutex> lock(mMutexConnections);
            if(mnID == 0)
                return;
            else if(mbNotErase)     //表示不该擦除该KeyFrame
            {
                mbToBeErased = true;
                return;
            }
        }
        
    }

} // namespace Goudan_SLAM