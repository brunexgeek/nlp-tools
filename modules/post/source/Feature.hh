#ifndef NLPTOOLS_POST_FEATURE_HH
#define NLPTOOLS_POST_FEATURE_HH


namespace nlptools {
namespace postagger {


class Feature
{
    public:
        Feature(
            const int label,
            const int feature )
        {
            content = (feature << 8) + (label & 0xff);
        }

        int label() const
        {
            return content & 0xff;
        }

        int feature() const
        {
            return content >> 8;
        }

        mefeature_type body() const
        {
            return content;
        }

    private:
        mefeature_type content;
};



struct FeatureBag
{
    #ifdef USE_HASH_MAP
    //    typedef __gnu_cxx::hash_map<mefeature_type, int> map_type;
    typedef std::tr1::unordered_map<mefeature_type, int> map_type;
    #else
    typedef std::map<mefeature_type, int> map_type;
    #endif
    map_type mef2id;
    std::vector<Feature> id2mef;

    int put(const Feature & i)
    {
        map_type::const_iterator j = mef2id.find(i.body());
        if (j == mef2id.end())
        {
            const int id = (int) id2mef.size();
            id2mef.push_back(i);
            mef2id[i.body()] = id;
            return id;
        }
        return j->second;
    }

    int getId(
        const Feature &code ) const
    {
        map_type::const_iterator j = mef2id.find(code.body());
        if (j == mef2id.end()) return -1;
        return j->second;
    }

    Feature getFeature(
        int id ) const
    {
        assert(id >= 0 && id < (int)id2mef.size());
        return id2mef[id];
    }

    int size() const
    {
        return (int) id2mef.size();
    }

    void clear()
    {
        mef2id.clear();
        id2mef.clear();
    }
};


}}


#endif // NLPTOOLS_POST_FEATURE_HH