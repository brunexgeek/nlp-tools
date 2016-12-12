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


// use this option and change MAX_LABEL_TYPES below accordingly, if you want to use more than 255 labels
//#define USE_MANY_LABELS

//
// data format for each sample for training/testing
//
struct CRF_State
{
    public:
        CRF_State() : label("") {};
        CRF_State(const std::string & l) : label(l) {};
        void set_label(const std::string & l) { label = l; }

        // to add a binary feature
        void add_feature(const std::string & f)
        {
            if (f.find_first_of('\t') != std::string::npos)
            {
                std::cerr << "error: illegal characters in a feature string" << std::endl;
                exit(1);
            }
            features.push_back(f);
        }

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
            const CRF_State & state )
        {
            content.push_back(state);
        }


        CRF_State &operator [](
            size_t index )
        {
            return content[index];
        }

        const CRF_State &operator [](
            size_t index ) const
        {
            return content[index];
        }

        size_t size() const
        {
            return content.size();
        }

        void push_back(
            const CRF_State &item )
        {
            content.push_back(item);
        }

        void pop_back()
        {
            content.pop_back();
        }

        iterator begin()
        {
            return content.begin();
        }

        iterator end()
        {
            return content.end();
        }

        const_iterator begin() const
        {
            return content.begin();
        }

        const_iterator end() const
        {
            return content.end();
        }

    private:
        std::vector<CRF_State> content;
};


typedef uint32_t mefeature_type;

class ME_Feature
{
    public:
        ME_Feature(
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



struct ME_FeatureBag
{
    #ifdef USE_HASH_MAP
    //    typedef __gnu_cxx::hash_map<mefeature_type, int> map_type;
    typedef std::tr1::unordered_map<mefeature_type, int> map_type;
    #else
    typedef std::map<mefeature_type, int> map_type;
    #endif
    map_type mef2id;
    std::vector<ME_Feature> id2mef;

    int put(const ME_Feature & i)
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
        const ME_Feature &code ) const
    {
        map_type::const_iterator j = mef2id.find(code.body());
        if (j == mef2id.end()) return -1;
        return j->second;
    }

    ME_Feature getFeature(
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


class CRF_Model
{
 public:

  enum OptimizationMethod { BFGS, PERCEPTRON, SGD };

  void add_training_sample(const CRF_Sequence & s);
  int train(const int cutoff = 0, const double sigma = 0, const double widthfactor = 0);
    void decode_lookahead(CRF_Sequence & s0);
  bool load_from_file(const std::string & filename, bool verbose = true);
  bool save_to_file(const std::string & filename, const double t = 0) const;
  int num_classes() const { return _num_classes; }
  std::string get_class_label(int i) const { return _label_bag.Str(i); }
  int get_class_id(const std::string & s) const { return _label_bag.Id(s); }
  void get_features(std::list< std::pair< std::pair<std::string, std::string>, double> > & fl);
  void set_heldout(const int h, const int n = 0) { _nheldout = h; _early_stopping_n = n; };
  //  bool load_from_array(const CRF_Model_Data data[]);

  enum { MAX_LABEL_TYPES = 50 };
  //  const static int MAX_LABEL_TYPES = 1000;
  enum { MAX_LEN = 1000 };

  void incr_line_counter() { _line_counter++; }

  CRF_Model();
  ~CRF_Model();

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
  StringBag _label_bag;
  //  MiniStringBag _featurename_bag;
  StrDic _featurename_bag;
  double _sigma; // Gaussian prior
  double _inequality_width;
  std::vector<double> _vl;  // vector of lambda
  std::vector<bool> is_edge;
  ME_FeatureBag _fb;
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
  double forward_backward(const Sequence & s);
  double viterbi(const Sequence & seq, std::vector<int> & best_seq);
  void initialize_edge_weights();
  void initialize_state_weights(const Sequence & seq);
  //  void lookahead_initialize_edge_weights();
  void lookahead_initialize_state_weights(const Sequence & seq);
  int make_feature_bag(const int cutoff);
  //  int classify(const Sample & nbs, std::vector<double> & membp) const;
  double update_model_expectation();
  double add_sample_model_expectation(const Sequence & seq, std::vector<double>& vme, int & ncorrect);
  void add_sample_empirical_expectation(const Sequence & seq, std::vector<double>& vee);
  int perform_BFGS();
  int perform_AveragedPerceptron();
  int perform_StochasticGradientDescent();
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
  //  std::vector<double> calc_state_weight(const Sequence & seq, const int i) const;
  std::vector<double> calc_state_weight(const int i) const;

  double FunctionGradient(const std::vector<double> & x, std::vector<double> & grad);
  static double FunctionGradientWrapper(const std::vector<double> & x, std::vector<double> & grad);

  int _line_counter; // for error message. Incremented at forward_backward

  int nbest_search_path[CRF_Model::MAX_LEN];

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


#endif


/*
 * $Log$
 */
