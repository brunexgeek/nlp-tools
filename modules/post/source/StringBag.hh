#ifndef NLPTOOLS_POST_STRINGBAG_HH
#define NLPTOOLS_POST_STRINGBAG_HH


namespace nlptools {
namespace postagger {


class MiniStringBag
{
    public:
        #ifdef USE_HASH_MAP
        typedef std::tr1::unordered_map<std::string, int> map_type;
        #else
        typedef std::map<std::string, int> map_type;
        #endif

        MiniStringBag() : _size(0) {}

        int Put(const std::string & i)
        {
            map_type::const_iterator j = str2id.find(i);
            if (j == str2id.end())
            {
                int id = _size;
                _size++;
                str2id[i] = id;
                return id;
            }
            return j->second;
        }

        int Id(const std::string & i) const
        {
            map_type::const_iterator j = str2id.find(i);
            if (j == str2id.end())  return -1;
            return j->second;
        }

        int size() const
        {
            return _size;
        }

        void clear()
        {
            str2id.clear(); _size = 0;
        }

        map_type::const_iterator begin() const { return str2id.begin(); }
        map_type::const_iterator end()   const { return str2id.end(); }

    protected:
        int _size;
        map_type str2id;
};

struct StringBag : public MiniStringBag
{
    std::vector<std::string> id2str;

    int Put(const std::string & i)
    {
        map_type::const_iterator j = str2id.find(i);
        if (j == str2id.end())
        {
            int id = (int) id2str.size();
            id2str.push_back(i);
            str2id[i] = id;
            return id;
        }
        return j->second;
    }

    std::string Str(const int id) const
    {
        assert(id >= 0 && id < (int)id2str.size());
        return id2str[id];
    }

    int size() const
    {
        return (int) id2str.size();
    }

    void clear()
    {
        str2id.clear();
        id2str.clear();
    }
};


}}

#endif // NLPTOOLS_POST_STRINGBAG_HH