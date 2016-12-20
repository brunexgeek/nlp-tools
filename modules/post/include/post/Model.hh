/*
 * $Id$
 */

#ifndef __CRF_H_
#define __CRF_H_

#include <string>
#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <iostream>
#include <string>
#include <cassert>
#include <cstdio>
#include <stdint.h>
#include <post/strdic.hh>
#include <post/common.hh>

#define USE_HASH_MAP  // if you encounter errors with hash, try commenting out this line. (the program will be a bit slower, though)
#ifdef USE_HASH_MAP
//#include <ext/hash_map>
#include <tr1/unordered_map>
#endif


namespace nlptools {
namespace postagger {


//
// data format for each sample for training/testing
//
struct CRF_State
{
    public:
        CRF_State();
        CRF_State(const std::string & l);
        void set_label(const std::string & l);

        // to add a binary feature
        void add_feature(const std::string & f);

    public:
        std::string label;
        std::vector<std::string> features;
};


struct CRF_Sequence
{
    public:
        typedef std::vector<CRF_State>::iterator iterator;
        typedef std::vector<CRF_State>::const_iterator const_iterator;

        void add_state(
            const CRF_State & state );


        CRF_State &operator [](
            size_t index );

        const CRF_State &operator [](
            size_t index ) const;

        size_t size() const;

        void push_back(
            const CRF_State &item );

        void pop_back();

        iterator begin();

        iterator end();

        const_iterator begin() const;

        const_iterator end() const;

    private:
        std::vector<CRF_State> content;
};


typedef uint32_t mefeature_type;



struct StringBag;


class FeatureBag;

/*
class NameTable : public std::map<std::string, int32_t>
{
    int32_t last;
    public:
        NameTable() : last(-1) {};

        ~NameTable() {};

        int32_t Put(const std::string &value )
        {
            insert( std::pair<std::string, int32_t>(value, last++) );
            return last;
        }

        int32_t Id(const std::string &value )
        {
            std::map<std::string, int32_t>::const_iterator it = find(value);
            if (it == end()) return -1;
            return it->second;
        }
};*/


class Model
{
    public:
        void add_training_sample(const CRF_Sequence & s);
        int train(const int cutoff = 0, const double sigma = 0, const double widthfactor = 0);
        void decode_lookahead(CRF_Sequence & s0);
        bool load_from_file(const std::string & filename, bool verbose = true);
        bool save_to_file(const std::string & filename, double t = 0) const;
        int num_classes() const { return _num_classes; }
        std::string get_class_label(int i) const;
        int get_class_id(const std::string & s) const;
        void get_features(std::list< std::pair< std::pair<std::string, std::string>, double> > & fl);
        void set_heldout(const int h, const int n = 0) { _nheldout = h; _early_stopping_n = n; };
        //  bool load_from_array(const CRF_Model_Data data[]);

        enum { MAX_LABEL_TYPES = 50 };
        //  const static int MAX_LABEL_TYPES = 1000;
        enum { MAX_LEN = 1000 };

        Model();
        ~Model();

    private:
        int lookaheadDepth;

        struct Sample {
        int label;
        std::vector<int> positive_features;
        };
        struct Sequence {
        std::vector<Sample> vs;
        };


        std::vector<Sequence> _vs; // vector of training_samples
        StringBag &_label_bag;
        StrDic _featurename_bag;
        //NameTable _featurename_bag;
        double _sigma; // Gaussian prior
        double _inequality_width;
        std::vector<double> _vl;  // vector of lambda
        std::vector<bool> is_edge;
        FeatureBag &_fb;
        int _num_classes;
        std::vector<double> _vee;  // empirical expectation
        std::vector<double> _vme;  // model expectation
        std::vector< std::vector< int > > _feature2mef;
        std::vector< Sequence > _heldout;
        double _train_error;   // current error rate on the training data
        double _heldout_error; // current error rate on the heldout data
        int _nheldout;
        int _early_stopping_n;
        std::vector<double> _vhlogl;

        double heldout_likelihood();
        double heldout_lookahead_error();
        void initialize_edge_weights();
        void initialize_state_weights(const Sequence & seq);
        void lookahead_initialize_state_weights(const Sequence & seq);
        int make_feature_bag(const int cutoff);
        double update_model_expectation();
        double add_sample_model_expectation(const Sequence & seq, std::vector<double>& vme, int & ncorrect);
        void add_sample_empirical_expectation(const Sequence & seq, std::vector<double>& vee);
        int perform_LookaheadTraining();

        double lookahead_search(const Sequence & seq,
                    std::vector<int> & history,
                    const int start,
                    const int max_depth,  const int depth,
                    double current_score,
                    std::vector<int> & best_seq,
                    const bool follow_gold = false,
                    const std::vector<int> *forbidden_seq = NULL);
        void calc_diff(const double val,
                const Sequence & seq,
                const int start,
                const std::vector<int> & history,
                const int depth, const int max_depth,
                std::map<int, double> & diff);
        int update_weights_sub(const Sequence & seq,
                    std::vector<int> & history,
                    const int x,
                    std::map<int, double> & diff);
        int update_weights_sub2(const Sequence & seq,
                    std::vector<int> & history,
                    const int x,
                    std::map<int, double> & diff);
        int update_weights_sub3(const Sequence & seq,
                    std::vector<int> & history,
                    const int x,
                    std::map<int, double> & diff);
        int lookaheadtrain_sentence(const Sequence & seq, int & t, std::vector<double> & wa);
        int decode_lookahead_sentence(const Sequence & seq, std::vector<int> & vs);

        void init_feature2mef();
        double calc_loglikelihood(const Sequence & seq);

        std::vector<double> calc_state_weight(const int i) const;

        double FunctionGradient(const std::vector<double> & x, std::vector<double> & grad);
        static double FunctionGradientWrapper(const std::vector<double> & x, std::vector<double> & grad);

        int nbest_search_path[Model::MAX_LEN];

        int *p_edge_feature_id;
        int *p_edge_feature_id2;
        int *p_edge_feature_id3;
        double *p_state_weight;
        double *p_edge_weight;
        double *p_edge_weight2;
        double *p_edge_weight3;
        double *p_forward_cache;
        double *p_backward_cache;
        int *p_backward_pointer;

        int & edge_feature_id3(const int w, const int x, const int y, const int z) const
            { assert(w >= 0 && w < MAX_LABEL_TYPES);
            assert(x >= 0 && x < MAX_LABEL_TYPES);
            assert(y >= 0 && y < MAX_LABEL_TYPES);
            assert(z >= 0 && z < MAX_LABEL_TYPES);
            return p_edge_feature_id3[w * MAX_LABEL_TYPES * MAX_LABEL_TYPES * MAX_LABEL_TYPES + x * MAX_LABEL_TYPES * MAX_LABEL_TYPES + y * MAX_LABEL_TYPES + z]; }
        int & edge_feature_id2(const int x, const int y, const int z) const
        { assert(x >= 0 && x < MAX_LABEL_TYPES);
            assert(y >= 0 && y < MAX_LABEL_TYPES);
            assert(z >= 0 && z < MAX_LABEL_TYPES);
            //      std::cout << x << " " << y << " " << z << std::endl;
            return p_edge_feature_id2[x * MAX_LABEL_TYPES * MAX_LABEL_TYPES + y * MAX_LABEL_TYPES + z]; }
        int & edge_feature_id(const int l, const int r) const
        { assert(l >= 0 && l < MAX_LABEL_TYPES);
            assert(r >= 0 && r < MAX_LABEL_TYPES);
            return p_edge_feature_id[l * MAX_LABEL_TYPES + r]; }
        double & state_weight(const int x, const int l) const
        { return p_state_weight[x * MAX_LABEL_TYPES + l]; }
        double & edge_weight2(const int x, const int y, const int z) const
        { return p_edge_weight2[x * MAX_LABEL_TYPES * MAX_LABEL_TYPES + y * MAX_LABEL_TYPES + z]; }
        double & edge_weight3(const int w, const int x, const int y, const int z) const
        { return p_edge_weight3[w * MAX_LABEL_TYPES * MAX_LABEL_TYPES * MAX_LABEL_TYPES + x * MAX_LABEL_TYPES * MAX_LABEL_TYPES + y * MAX_LABEL_TYPES + z]; }
        double & edge_weight(const int l, const int r) const
        { return p_edge_weight[l * MAX_LABEL_TYPES + r]; }
        double & forward_cache(const int x, const int l) const
        { return p_forward_cache[x * MAX_LABEL_TYPES + l]; }
        double & backward_cache(const int x, const int l) const
        { return p_backward_cache[x * MAX_LABEL_TYPES + l]; }
        int & backward_pointer(const int x, const int l) const
        { return p_backward_pointer[x * MAX_LABEL_TYPES + l]; }


};



typedef void FeatureCallback(
    const Sentence &sentence,
    int i,
    CRF_State &current );

}}


#endif

