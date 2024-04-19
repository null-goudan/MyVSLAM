#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "ORBExtractor.h"
#include <thread>

namespace Goudan_SLAM
{
    Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
    {
        mK = ReferenceFrame.mK.clone();
        mvKeys1 = ReferenceFrame.mvKeysUn;

        mSigma = sigma;
        mSigma2 = sigma * sigma;
        mMaxIterations = iterations;
    }

    bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                                 cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
    {
        std::cout << "Initializing..."<<endl;
        // Frame2 特征点
        mvKeys2 = CurrentFrame.mvKeysUn;

        // mvMatches12 记录匹配上的特征点对
        mvMatches12.clear();
        mvMatches12.reserve(mvKeys2.size());
        // mvbMatched1 记录每个特征点是否有匹配的特征点
        mvbMatches1.resize(mvKeys1.size());

        // 1. 组织特征点对
        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            if (vMatches12[i] > 0)
            {
                mvMatches12.push_back(make_pair(i, vMatches12[i]));
                mvbMatches1[i] = true;
            }
            else
                mvbMatches1[i] = false;
        }

        // 匹配上的特征点的个数
        const int N = mvMatches12.size();

        // 新建一个容器vAllIndices, 生成0 到 N-1 的数作为特征点的索引
        vector<size_t> vAllIndices;
        vAllIndices.reserve(N);
        vector<size_t> vAvailableIndices;

        for (int i = 0; i < N; i++)
        {
            vAllIndices.push_back(i);
        }

        // 2. 在所有匹配特征点钟随机选择8对匹配特征点一组
        // 共选择 mMaxIterations 组
        // 用于解F模型和H模型 分别是基础矩阵和单应矩阵
        // mMaxIterations: 200
        mvSets = vector<vector<size_t>>(mMaxIterations, vector<size_t>(8, 0));

        DUtils::Random::SeedRandOnce(0);

        for (int it = 0; it < mMaxIterations; it++)
        {
            vAvailableIndices = vAllIndices;

            for (size_t j = 0; j < 8; j++)
            {
                // 产生0到N-1的随机数
                int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
                // idx表示哪一个索引对应的特征点beixuanzhong
                int idx = vAvailableIndices[randi];

                mvSets[it][j] = idx;

                // randi 对应的索引被选过,就从容器中删除
                // randi 对应的索引用最后一个元素替换，然后删除最后一个元素
                vAvailableIndices[randi] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }
        }

        // 3. 启动多线程分别用于计算fundamental matrix 和 homography
        vector<bool> vbMatchesInliersH, vbMatchesInliersF;
        float SH, SF; // Score for H and F
        cv::Mat H, F;

        // 计算单应矩阵并打分
        thread threadH(&Initializer::FindHomography, this, ref(vbMatchesInliersH), ref(SH), ref(H));
        // 计算基础矩阵并打分
        thread threadF(&Initializer::FindFundamental, this, ref(vbMatchesInliersF), ref(SF), ref(F));

        threadH.join();
        threadF.join();

        // 4. 计算得分比例，选取某个模型
        float RH = SH / (SH + SF);

        std::cout<<"initialize success. reconstructing R and t ..." <<endl;
        // 5. 从H矩阵或者F矩阵中恢复R,t
        if (RH > 0.40){
            cout << "select Homography model" <<endl;
            return ReconstructH(vbMatchesInliersH, H, mK, R21, t21, vP3D, vbTriangulated, 0, 50);
        }
        else{
            cout << "select Fundamanetal model" <<endl;
            return ReconstructF(vbMatchesInliersF, F, mK, R21, t21, vP3D, vbTriangulated, 0, 50);
        }
        return false;
    }

    // 计算单应矩阵
    void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
    {
        const int N = mvMatches12.size();

        // 将mvKeys1和mvKeys2归一化到均值为0， 一阶绝对矩为1， 归一化矩阵分别为T1, T2
        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(mvKeys1, vPn1, T1);
        Normalize(mvKeys2, vPn2, T2);
        cv::Mat T2inv = T2.inv();

        // 最终最佳的MatchesInliner和得分
        score = 0.0;
        vbMatchesInliers = vector<bool>(N, false);

        // Iteration variables
        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);

        cv::Mat H21i, H12i;
        // 每次RANSAC的MatchesInliners与得分
        vector<bool> vbCurrentInliners(N, false);
        float currentScore;

        // 运行RANSIC循环并保存最高分数的解决
        for (int it = 0; it < mMaxIterations; it++)
        {
            // 八点法(但实际上最少4个匹配点就够)
            for (size_t j = 0; j < 8; j++)
            {
                int idx = mvSets[it][j];

                // vPn1 和 vPn2i 为匹配的特征点对的坐标
                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            cv::Mat Hn = ComputeH21(vPn1i, vPn2i);
            // 恢复原始的均值和尺度
            H21i = T2inv * Hn * T1;
            H12i = H21i.inv();

            // 利用重投影误差为当次RANSAC的结果评分
            currentScore = CheckHomography(H21i, H12i, vbCurrentInliners, mSigma);

            // 得到最优的vbMatcherInliners与Score
            if (currentScore > score)
            {
                H21 = H21i.clone();
                vbMatchesInliers = vbCurrentInliners;
                score = currentScore;
            }
        }
    }

    void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
    {
        const int N = vbMatchesInliers.size();

        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(mvKeys1, vPn1, T1);
        Normalize(mvKeys2, vPn2, T2);
        cv::Mat T2t = T2.t();

        score = 0.0;
        vbMatchesInliers = vector<bool>(N, false);

        // Iteration variables
        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);
        cv::Mat F21i;
        vector<bool> vbCurrentInliers(N, false);
        float currentScore;

        for (int it = 0; it < mMaxIterations; it++)
        {
            // Select a minimum set
            for (int j = 0; j < 8; j++)
            {
                int idx = mvSets[it][j];

                vPn1i[j] = vPn1[mvMatches12[idx].first];
                vPn2i[j] = vPn2[mvMatches12[idx].second];
            }

            cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

            F21i = T2t * Fn * T1;

            // 利用重投影误差为当次RANSAC的结果评分
            currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

            if (currentScore > score)
            {
                F21 = F21i.clone();
                vbMatchesInliers = vbCurrentInliers;
                score = currentScore;
            }
        }
    }

    // 将特征点 (u,v,1) 认为是3D点进一步投影到虚拟图像上， 解决特征点分布不均匀的情况
    void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
    {
        float meanX = 0;
        float meanY = 0;
        const int N = vKeys.size();

        vNormalizedPoints.resize(N);

        for (int i = 0; i < N; i++)
        {
            meanX += vKeys[i].pt.x;
            meanY += vKeys[i].pt.y;
        }

        meanX = meanX / N;
        meanY = meanY / N;

        float meanDevX = 0;
        float meanDevY = 0;

        // 将所有点的坐标减去中心坐标，使其坐标均值为0
        for (int i = 0; i < N; i++)
        {
            vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
            vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

            meanDevX += fabs(vNormalizedPoints[i].x);
            meanDevY += fabs(vNormalizedPoints[i].y);
        }

        meanDevX = meanDevX / N;
        meanDevY = meanDevY / N;

        // 尺度计算
        float sX = 1.0 / meanDevX;
        float sY = 1.0 / meanDevY;

        // 将坐标尺度缩放， 使得一阶绝对矩分别为1
        for (int i = 0; i < N; i++)
        {
            vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
            vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
        }

        // 计算均一化的矩阵
        // |sX  0  -meanx*sX|
        // |0   sY -meany*sY|
        // |0   0      1    |
        T = cv::Mat::eye(3, 3, CV_32F);
        T.at<float>(0, 0) = sX;
        T.at<float>(1, 1) = sY;
        T.at<float>(0, 2) = -meanX * sX;
        T.at<float>(1, 2) = -meanY * sY;
    }

    cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
    {
        const int N = vP1.size();

        cv::Mat A(2 * N, 9, CV_32F); // 2N*9

        for (int i = 0; i < N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            A.at<float>(2 * i, 0) = 0.0;
            A.at<float>(2 * i, 1) = 0.0;
            A.at<float>(2 * i, 2) = 0.0;
            A.at<float>(2 * i, 3) = -u1;
            A.at<float>(2 * i, 4) = -v1;
            A.at<float>(2 * i, 5) = -1;
            A.at<float>(2 * i, 6) = v2 * u1;
            A.at<float>(2 * i, 7) = v2 * v1;
            A.at<float>(2 * i, 8) = v2;

            A.at<float>(2 * i + 1, 0) = u1;
            A.at<float>(2 * i + 1, 1) = v1;
            A.at<float>(2 * i + 1, 2) = 1;
            A.at<float>(2 * i + 1, 3) = 0.0;
            A.at<float>(2 * i + 1, 4) = 0.0;
            A.at<float>(2 * i + 1, 5) = 0.0;
            A.at<float>(2 * i + 1, 6) = -u2 * u1;
            A.at<float>(2 * i + 1, 7) = -u2 * v1;
            A.at<float>(2 * i + 1, 8) = -u2;
        }

        cv::Mat u, w, vt;

        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        return vt.row(8).reshape(0, 3); // v的最后一列即为Homography矩阵
    }

    cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
    {
        const int N = vP1.size();

        cv::Mat A(N, 9, CV_32F);

        for (int i = 0; i < N; i++)
        {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            // [u2u1, u2v1, u2, v2u1, v2v1, v2, u1, v1]*e = 0
            // e 为本质矩阵展开的向量
            A.at<float>(i, 0) = u2 * u1;
            A.at<float>(i, 1) = u2 * v1;
            A.at<float>(i, 2) = u2;
            A.at<float>(i, 3) = v2 * u1;
            A.at<float>(i, 4) = v2 * v1;
            A.at<float>(i, 5) = v2;
            A.at<float>(i, 6) = u1;
            A.at<float>(i, 7) = v1;
            A.at<float>(i, 8) = 1;
        }

        // 奇异值分解 E=UWV(-1) U V正交阵 W 奇异特征值阵 有秩2约束
        cv::Mat u, w, vt;
        cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        cv::Mat Fpre = vt.row(8).reshape(0, 3); // v的最后一列 (应该是意味着最后一个解E)
        cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        w.at<float>(2) = 0;               // 秩2约束，将第三个奇异值设为0
        return u * cv::Mat::diag(w) * vt; // Fundamental矩阵
    }

    float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat H12, vector<bool> vbMatchesInliners, float sigma)
    {
        const int N = mvMatches12.size();

        // |h11 h12 h13|
        // |h21 h22 h23|
        // |h31 h32 h33|
        const float h11 = H21.at<float>(0, 0);
        const float h12 = H21.at<float>(0, 1);
        const float h13 = H21.at<float>(0, 2);
        const float h21 = H21.at<float>(1, 0);
        const float h22 = H21.at<float>(1, 1);
        const float h23 = H21.at<float>(1, 2);
        const float h31 = H21.at<float>(2, 0);
        const float h32 = H21.at<float>(2, 1);
        const float h33 = H21.at<float>(2, 2);

        // |h11inv h12inv h13inv|
        // |h21inv h22inv h23inv|
        // |h31inv h32inv h33inv|
        const float h11inv = H12.at<float>(0, 0);
        const float h12inv = H12.at<float>(0, 1);
        const float h13inv = H12.at<float>(0, 2);
        const float h21inv = H12.at<float>(1, 0);
        const float h22inv = H12.at<float>(1, 1);
        const float h23inv = H12.at<float>(1, 2);
        const float h31inv = H12.at<float>(2, 0);
        const float h32inv = H12.at<float>(2, 1);
        const float h33inv = H12.at<float>(2, 2);

        vbMatchesInliners.resize(N);

        float score = 0;

        // 基于卡方检验计算出的阈值(假设测量有一个像素的偏差)
        const float th = 5.991;
        // 信息矩阵， 方差平方的倒数
        const float invSigmaSquare = 1.0 / (sigma * sigma);
        // N对特征匹配点
        for (int i = 0; i < N; i++)
        {
            bool bIn = true;

            const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].first];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            // 重投影在第一个图像中（即将图像2的点单应到图像1中）
            // |u1|   |h11inv h12inv h13inv||u2|
            // |v1| = |h21inv h22inv h23inv||v2|
            // |1 |   |h31inv h32inv h33inv||1 |
            const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
            const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
            const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

            // 计算重投影误差
            const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

            // 根据方差归一化误差
            const float chiSquare1 = squareDist1 * invSigmaSquare;

            if (chiSquare1 > th)
                bIn = false;
            else
                score += th - chiSquare1;

            // 将图像1的特征点单应在图像2 中
            const float w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
            const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
            const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

            const float squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

            const float chiSquare2 = squareDist2 * invSigmaSquare;

            if (chiSquare2 > th)
                bIn = false;
            else
                score += th - chiSquare2;

            if (bIn)
                vbMatchesInliners[i] = true;
            else
                vbMatchesInliners[i] = false;
        }
        return score;
    }

    // 使用对极约束中的极线
    float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &bvMatcherInliners, float sigma)
    {
        const int N = mvMatches12.size();

        const float f11 = F21.at<float>(0, 0);
        const float f12 = F21.at<float>(0, 1);
        const float f13 = F21.at<float>(0, 2);
        const float f21 = F21.at<float>(1, 0);
        const float f22 = F21.at<float>(1, 1);
        const float f23 = F21.at<float>(1, 2);
        const float f31 = F21.at<float>(2, 0);
        const float f32 = F21.at<float>(2, 1);
        const float f33 = F21.at<float>(2, 2);

        bvMatcherInliners.resize(N);

        float score = 0;
        // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
        const float th = 3.841;
        const float thScore = 5.991;

        const float invSigmaSquare = 1.0 / (sigma * sigma);

        for (int i = 0; i < N; i++)
        {
            bool bIn = true;

            const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
            const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

            const float u1 = kp1.pt.x;
            const float v1 = kp1.pt.y;
            const float u2 = kp2.pt.x;
            const float v2 = kp2.pt.y;

            // Reprojection error in second image
            // l2=F21x1=(a2,b2,c2)
            // F21x1可以算出x1在图像中x2对应的线l
            const float a2 = f11 * u1 + f12 * v1 + f13;
            const float b2 = f21 * u1 + f22 * v1 + f23;
            const float c2 = f31 * u1 + f32 * v1 + f33;

            // x2应该在l这条线上:x2点乘l = 0
            const float num2 = a2 * u2 + b2 * v2 + c2;

            const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2); // 点到线的几何距离 的平方

            const float chiSquare1 = squareDist1 * invSigmaSquare;

            if (chiSquare1 > th)
                bIn = false;
            else
                score += thScore - chiSquare1;

            // Reprojection error in second image
            // l1 =x2tF21=(a1,b1,c1)

            const float a1 = f11 * u2 + f21 * v2 + f31;
            const float b1 = f12 * u2 + f22 * v2 + f32;
            const float c1 = f13 * u2 + f23 * v2 + f33;

            const float num1 = a1 * u1 + b1 * v1 + c1;

            const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

            const float chiSquare2 = squareDist2 * invSigmaSquare;

            if (chiSquare2 > th)
                bIn = false;
            else
                score += thScore - chiSquare2;

            if (bIn)
                bvMatcherInliners[i] = true;
            else
                bvMatcherInliners[i] = false;
        }

        return score;
    }

    // :TOREAD ! 
    // H矩阵分解常见有两种方法：Faugeras SVD-based decomposition 和 Zhang SVD-based decomposition
    // 参考文献：Motion and structure from motion in a piecewise plannar environment
    // 这篇参考文献和下面的代码使用了Faugeras SVD-based decomposition算法

    /**
     * @brief 从H恢复R t
     *
     * @see
     * - Faugeras et al, Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988.
     * - Deeper understanding of the homography decomposition for vision-based control
     */
    bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                                   cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
    {
        int N = 0;
        for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
            if (vbMatchesInliers[i])
                N++;

        // We recover 8 motion hypotheses using the method of Faugeras et al.
        // Motion and structure from motion in a piecewise planar environment.
        // International Journal of Pattern Recognition and Artificial Intelligence, 1988

        // step1：SVD分解Homography
        // 因为特征点是图像坐标系，所以讲H矩阵由相机坐标系换算到图像坐标系
        cv::Mat invK = K.inv();
        cv::Mat A = invK * H21 * K;

        cv::Mat U, w, Vt, V;
        cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);
        V = Vt.t();

        float s = cv::determinant(U) * cv::determinant(Vt);

        float d1 = w.at<float>(0);
        float d2 = w.at<float>(1);
        float d3 = w.at<float>(2);

        // SVD分解的正常情况是特征值降序排列
        if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001)
        {
            return false;
        }

        vector<cv::Mat> vR, vt, vn;
        vR.reserve(8);
        vt.reserve(8);
        vn.reserve(8);

        // step2：计算法向量
        // n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
        // 法向量n'= [x1 0 x3] 对应ppt的公式17
        float aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
        float aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
        float x1[] = {aux1, aux1, -aux1, -aux1};
        float x3[] = {aux3, -aux3, aux3, -aux3};

        // step3：恢复旋转矩阵
        // step3.1：计算 sin(theta)和cos(theta)，case d'=d2
        // 计算ppt中公式19
        float aux_stheta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

        float ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
        float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

        // step3.2：计算四种旋转矩阵R，t
        // 计算旋转矩阵 R‘，计算ppt中公式18
        //      | ctheta      0   -aux_stheta|       | aux1|
        // Rp = |    0        1       0      |  tp = |  0  |
        //      | aux_stheta  0    ctheta    |       |-aux3|

        //      | ctheta      0    aux_stheta|       | aux1|
        // Rp = |    0        1       0      |  tp = |  0  |
        //      |-aux_stheta  0    ctheta    |       | aux3|

        //      | ctheta      0    aux_stheta|       |-aux1|
        // Rp = |    0        1       0      |  tp = |  0  |
        //      |-aux_stheta  0    ctheta    |       |-aux3|

        //      | ctheta      0   -aux_stheta|       |-aux1|
        // Rp = |    0        1       0      |  tp = |  0  |
        //      | aux_stheta  0    ctheta    |       | aux3|
        for (int i = 0; i < 4; i++)
        {
            cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
            Rp.at<float>(0, 0) = ctheta;
            Rp.at<float>(0, 2) = -stheta[i];
            Rp.at<float>(2, 0) = stheta[i];
            Rp.at<float>(2, 2) = ctheta;

            cv::Mat R = s * U * Rp * Vt;
            vR.push_back(R);

            cv::Mat tp(3, 1, CV_32F);
            tp.at<float>(0) = x1[i];
            tp.at<float>(1) = 0;
            tp.at<float>(2) = -x3[i];
            tp *= d1 - d3;

            // 这里虽然对t有归一化，并没有决定单目整个SLAM过程的尺度
            // 因为CreateInitialMapMonocular函数对3D点深度会缩放，然后反过来对 t 有改变
            cv::Mat t = U * tp;
            vt.push_back(t / cv::norm(t));

            cv::Mat np(3, 1, CV_32F);
            np.at<float>(0) = x1[i];
            np.at<float>(1) = 0;
            np.at<float>(2) = x3[i];

            cv::Mat n = V * np;
            if (n.at<float>(2) < 0)
                n = -n;
            vn.push_back(n);
        }

        // step3.3：计算 sin(theta)和cos(theta)，case d'=-d2
        // 计算ppt中公式22
        float aux_sphi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

        float cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
        float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

        // step3.4：计算四种旋转矩阵R，t
        // 计算旋转矩阵 R‘，计算ppt中公式21
        for (int i = 0; i < 4; i++)
        {
            cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
            Rp.at<float>(0, 0) = cphi;
            Rp.at<float>(0, 2) = sphi[i];
            Rp.at<float>(1, 1) = -1;
            Rp.at<float>(2, 0) = sphi[i];
            Rp.at<float>(2, 2) = -cphi;

            cv::Mat R = s * U * Rp * Vt;
            vR.push_back(R);

            cv::Mat tp(3, 1, CV_32F);
            tp.at<float>(0) = x1[i];
            tp.at<float>(1) = 0;
            tp.at<float>(2) = x3[i];
            tp *= d1 + d3;

            cv::Mat t = U * tp;
            vt.push_back(t / cv::norm(t));

            cv::Mat np(3, 1, CV_32F);
            np.at<float>(0) = x1[i];
            np.at<float>(1) = 0;
            np.at<float>(2) = x3[i];

            cv::Mat n = V * np;
            if (n.at<float>(2) < 0)
                n = -n;
            vn.push_back(n);
        }

        int bestGood = 0;
        int secondBestGood = 0;
        int bestSolutionIdx = -1;
        float bestParallax = -1;
        vector<cv::Point3f> bestP3D;
        vector<bool> bestTriangulated;

        // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
        // We reconstruct all hypotheses and check in terms of triangulated points and parallax
        // step4：d'=d2和d'=-d2分别对应8组(R t)，通过恢复3D点并判断是否在相机正前方的方法来确定最优解
        for (size_t i = 0; i < 8; i++)
        {
            float parallaxi;
            vector<cv::Point3f> vP3Di;
            vector<bool> vbTriangulatedi;
            int nGood = CheckRT(vR[i], vt[i], mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3Di, 4.0 * mSigma2, vbTriangulatedi, parallaxi);

            // 保留最优的和次优的
            if (nGood > bestGood)
            {
                secondBestGood = bestGood;
                bestGood = nGood;
                bestSolutionIdx = i;
                bestParallax = parallaxi;
                bestP3D = vP3Di;
                bestTriangulated = vbTriangulatedi;
            }
            else if (nGood > secondBestGood)
            {
                secondBestGood = nGood;
            }
        }

        // step5：通过判断最优是否明显好于次优，从而判断该次Homography分解是否成功
        if (secondBestGood < 0.75 * bestGood && bestParallax >= minParallax && bestGood > minTriangulated && bestGood > 0.9 * N)
        {
            vR[bestSolutionIdx].copyTo(R21);
            vt[bestSolutionIdx].copyTo(t21);
            vP3D = bestP3D;
            vbTriangulated = bestTriangulated;

            return true;
        }

        return false;
    }

    /**
     * @brief 从F恢复R t
     *
     * 度量重构
     * 1. 由Fundamental矩阵结合相机内参K，得到Essential矩阵: \f$ E = k'^T F k \f$
     * 2. SVD分解得到R t
     * 3. 进行cheirality check, 从四个解中找出最合适的解
     *
     * @see Multiple View Geometry in Computer Vision <<多视图几何>> - Result 9.19 p259
     */
    bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                                   cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
    {
        int N = 0;
        for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
            if (vbMatchesInliers[i])
                N++;

        cout << "Compute Essencial Matrix from Fundamental.." <<endl;
        // 通过Fundamental矩阵计算Esencial矩阵
        cv::Mat E21 = K.t() * F21 * K;

        cv::Mat R1, R2, t;

        cout << "Decompose essencial matrix .."<<endl;
        // 通过分解得到四个模型然后筛选
        DecomposeE(E21, R1, R2, t);

        cv::Mat t1 = t;
        cv::Mat t2 = -t;

        cout << "reconstruct with the 4 hyphoteses and check .."<<endl;
        // Reconstruct with the 4 hyphoteses and check
        vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
        vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
        float parallax1, parallax2, parallax3, parallax4;

        int nGood1 = CheckRT(R1, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D1, 4.0 * mSigma2, vbTriangulated1, parallax1);
        int nGood2 = CheckRT(R2, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D2, 4.0 * mSigma2, vbTriangulated2, parallax2);
        int nGood3 = CheckRT(R1, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D3, 4.0 * mSigma2, vbTriangulated3, parallax3);
        int nGood4 = CheckRT(R2, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers, K, vP3D4, 4.0 * mSigma2, vbTriangulated4, parallax4);

        int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));

        R21 = cv::Mat();
        t21 = cv::Mat();

        // minTriangulated 为可以三角化恢复三维点的个数
        int nMinGood = max(static_cast<int>(0.9 * N), minTriangulated);

        int nsimilar = 0;
        if (nGood1 > 0.7 * maxGood)
            nsimilar++;
        if (nGood2 > 0.7 * maxGood)
            nsimilar++;
        if (nGood3 > 0.7 * maxGood)
            nsimilar++;
        if (nGood4 > 0.7 * maxGood)
            nsimilar++;

        // If there is not a clear winner or not enough triangulated points reject initialization
        // 四个结果中如果没有明显的最优结果，则返回失败
        if (maxGood < nMinGood || nsimilar > 1)
        {
            return false;
        }

        // If best reconstruction has enough parallax initialize
        // 比较大的视差角
        if (maxGood == nGood1)
        {
            if (parallax1 > minParallax)
            {
                vP3D = vP3D1;
                vbTriangulated = vbTriangulated1;

                R1.copyTo(R21);
                t1.copyTo(t21);
                return true;
            }
        }
        else if (maxGood == nGood2)
        {
            if (parallax2 > minParallax)
            {
                vP3D = vP3D2;
                vbTriangulated = vbTriangulated2;

                R2.copyTo(R21);
                t1.copyTo(t21);
                return true;
            }
        }
        else if (maxGood == nGood3)
        {
            if (parallax3 > minParallax)
            {
                vP3D = vP3D3;
                vbTriangulated = vbTriangulated3;

                R1.copyTo(R21);
                t2.copyTo(t21);
                return true;
            }
        }
        else if (maxGood == nGood4)
        {
            if (parallax4 > minParallax)
            {
                vP3D = vP3D4;
                vbTriangulated = vbTriangulated4;

                R2.copyTo(R21);
                t2.copyTo(t21);
                return true;
            }
        }

        return false;
    }

    // 分解Essential矩阵
    // F矩阵通过结合内参可以得到Essential矩阵，分解E矩阵可以得到4组解
    // 见 <<多视图几何>> 9.19 p259
    void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
    {
        cv::Mat u, w, vt;
        cv::SVD::compute(E, w, u, vt);

        u.col(2).copyTo(t);
        t = t / cv::norm(t);

        cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
        W.at<float>(0, 1) = -1;
        W.at<float>(1, 0) = 1;
        W.at<float>(2, 2) = 1;

        R1 = u * W * vt;
        if (cv::determinant(R1) < 0) //  旋转矩阵有行列式为1的约束
            R1 = -R1;

        R2 = u * W.t() * vt;
        if (cv::determinant(R2) < 0)
            R2 = -R2;
    }

    // 进行cheirality check，从而进一步找出F分解后最合适的解
    int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                             const vector<Match> &vMatches12, vector<bool> &vbMatchesInlies,
                             const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
    {
        // 相机内参
        const float fx = K.at<float>(0, 0);
        const float fy = K.at<float>(1, 1);
        const float cx = K.at<float>(0, 2);
        const float cy = K.at<float>(1, 2);

        vbGood = vector<bool>(vKeys1.size(), false);
        vP3D.resize(vKeys1.size());

        vector<float> vCosParallax;
        vCosParallax.reserve(vKeys1.size());

        // 1.得到一个相机的投影矩阵
        //  以第一个相机的广信作为世界坐标系
        cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
        K.copyTo(P1.rowRange(0, 3).colRange(0, 3));
        // 第一个相机的光心在世界坐标系下的坐标
        cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

        // 2. 得到第二个相机的投影矩阵
        cv::Mat P2(3, 4, CV_32F);
        R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
        t.copyTo(P2.rowRange(0, 3).col(3));
        P2 = K * P2;
        // 第二个相机在世界坐标系下的坐标
        cv::Mat O2 = -R.t() * t;

        int nGood = 0;
        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            if (!vbMatchesInlies[i])
                continue;
            // kp1 和 kp2 是匹配特征点
            const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
            const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
            cv::Mat p3dC1;

            // 3. 用三角法回复三维点p3dC1
            Triangulate(kp1, kp2, P1, P2, p3dC1);

            if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
            {
                // 如果点无穷大就是废点
                vbGood[vMatches12[i].first] = false;
                continue;
            }

            // 检查是否平行（视差角大小）
            //  4.计算视差角余弦值
            cv::Mat normal1 = p3dC1 - O1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = p3dC1 - O2;
            float dist2 = cv::norm(normal2);

            float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

            // 5. 判断3D点是否在两个摄像头前方
            // 5.1  3D点深度为负，在第一个摄像头后方，淘汰
            if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998)
                continue;

            cv::Mat p3dC2 = R * p3dC1 + t;
            // 5.2  3D点深度为负，在第二个摄像头后方，淘汰
            if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998)
                continue;

            // 6. 计算重投影误差
            // 计算3D点在第一个图像上的误差
            float im1x, im1y;
            float invZ1 = 1.0 / p3dC1.at<float>(2);
            im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
            im1y = fy * p3dC2.at<float>(1) * invZ1 + cy;

            float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

            // 6.1 重投影误差太大，直接淘汰然后跳过
            // 一般视差角比较小的时候重投影误差比较大
            if (squareError1 > th2)
                continue;

            // 计算3D点在第二个图像上的投影误差
            float im2x, im2y;
            float invZ2 = 1.0 / p3dC2.at<float>(2);
            im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
            im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;

            float squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

            // 步骤6.2：重投影误差太大，跳过淘汰
            // 一般视差角比较小时重投影误差比较大
            if (squareError2 > th2)
                continue;

            // 7. 统计经过检验的3D个数， 记录3D点视差角
            vCosParallax.push_back(cosParallax);
            vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
            nGood++;

            if (cosParallax < 0.99998)
                vbGood[vMatches12[i].first] = true;
        }
        // 步骤8：得到3D点中较大的视差角
        if (nGood > 0)
        {
            // 从小到大排序
            sort(vCosParallax.begin(), vCosParallax.end());

            // trick! 排序后并没有取最大的视差角
            // 取一个较大的视差角
            size_t idx = min(50, int(vCosParallax.size() - 1));
            parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
        }
        else
            parallax = 0;

        return nGood;
    }

    // 给定投影矩阵P1，P2和图像上的点kp1,kp2，从而恢复3D坐标
    // 三角化算法 <<多视图几何>>
    void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
    {
        cv::Mat A(4, 4, CV_32F);

        A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
        A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
        A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
        A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

        cv::Mat u, w, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        x3D = vt.row(3).t();
        x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
    }
}