#include "System.h"

using namespace std;

const string strPathToSequence = "./dataset/image_00";
const string settingFilePath = "./KITTI00-02.yaml";

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main()
{
    // 处理路径加载图像
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(strPathToSequence, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    Goudan_SLAM::System SLAM(settingFilePath);

    // 统计时间的Vector
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // 主循环
    cv::Mat im;
    // cv::namedWindow("dataset", cv::WINDOW_AUTOSIZE );
    for(int ni = 0; ni<nImages; ni++){
        // Read image from file
        cout << vstrImageFilenames[ni] <<endl;
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }
        // Tracking
        SLAM.TrackMonocular(im, tframe);
        // :TODO
        // cv::imshow("dataset", im);
        // cv::waitKey(10);
    }

    return 0;
}

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/timestamps.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof()){
        string s;
        getline(fTimes, s);
        if(!s.empty()){
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            
            vTimestamps.push_back(t);
        }
    }
    string strPrefixLeft = strPathToSequence + "/data/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(10) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
        // cout << vstrImageFilenames[i] << endl; // test;
    }
}