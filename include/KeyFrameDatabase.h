#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>

#include "KeyFrame.h"
#include "Frame.h"
#include "ORBVocabulary.h"

#include <mutex>

namespace Goudan_SLAM
{
    class KeyFrame;
    class Frame;

    class KeyFrameDatabase
    {
    public:
        KeyFrameDatabase(const ORBVocabulary &Voc);
        void add(KeyFrame* pKF);
        void erase(KeyFrame* pKF);
        void clear();

        // Loop Detection
        // std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);

        // Relocalization
    protected:
        const ORBVocabulary* mpVoc;

        std::vector<list<KeyFrame*>> mvInvertedFile;

        // Mutex
        std::mutex mMutex;
    };
}

#endif