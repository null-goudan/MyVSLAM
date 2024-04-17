#include "Initializer.h"

#include "ORBExtractor.h"
#include <thread>


namespace Goudan_SLAM{
    Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations){
        mK = ReferenceFrame.mK.clone();
        mvKeys1 = ReferenceFrame.mvKeysUn;

        mSigma = sigma;
        mSigma2 = sigma * sigma;
        mMaxIterations = iterations;
    }

    bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
    {
        // Frame2 特征点
        mvKeys2 = CurrentFrame.mvKeysUn;

        // mvMatches12 记录匹配上的特征点对
        mvMatches12.clear();
        mvMatches12.reserve(mvKeys2.size());
        // mvbMatched1 记录每个特征点是否有匹配的特征点
        mvbMatches1.resize(mvKeys1.size());

        // 1. 组织特征点对
        for(size_t i = 0,iend=vMatches12.size(); i < iend; i++){
            if(vMatches12[i] > 0){
                mvMatches12.push_back(make_pair(i, vMatches12[i]));
                mvbMatches1[i] = true;
            }else
                mvbMatches1[i] = false;
        }

        // 匹配上的特征点的个数
        const int N = mvMatches12.size();

        // 新建一个容器vAllIndices, 生成0 到 N-1 的数作为特征点的索引
        vector<size_t> vAllIndices;
        vAllIndices.reserve(N);
        vector<size_t> vAvailableIndices; 

        for(int i = 0; i<N; i++){
            vAllIndices.push_back(i);
        }

        // 2. 在所有匹配特征点钟随机选择8对匹配特征点一组
        // 共选择 mMaxIterations 组
        // 用于解F模型和H模型 分别是基础矩阵和单应矩阵
        // mMaxIterations: 200
        mvSets = vector< vector<size_t> > (mMaxIterations, vector<size_t>(8,0));

        
        
    }

}