/*
 * $Id$
 */

#include <sys/time.h>
#include <stdio.h>
#include <fstream>
#include <map>
#include <list>
#include <iostream>
#include <sstream>
#include <cmath>
#include <set>
#include <post/Trainer.hh>
#include <post/crf.hh>
#include <post/common.hh>

using namespace std;

#define CALLED_THIS      //std::cout << __FUNCTION__ << std::endl;

multimap<string, string> WNdic;

//extern string normalize(const string & s);
void tokenize(const string & s1, list<string> & lt);
string base_form(const string & s, const string & pos);

extern int push_stop_watch();


namespace nlptools {
namespace postagger {


static string
normalize(const string & s)
{
    CALLED_THIS;
    string tmp(s);
    for (size_t i = 0; i < tmp.size(); i++)
    {
        if (tmp[i] >= 65 && tmp[i] <= 90)
          tmp[i] += 32;
        else
        if (tmp[i] >= 192 && tmp[i] <= 221)
          tmp[i] += 32;
        else
        if (isdigit(tmp[i])) tmp[i] = '#';
    }
    //if (tmp[tmp.size()-1] == 's') tmp = tmp.substr(0, tmp.size()-1);
    return tmp;
}

//--------------------------------------------------------------------
// If you want to use stepp as a chunker, use this function instead
// of the original crfstate().
// Also, make sure that you use -f option both in training and testing
//--------------------------------------------------------------------
/*
static CRF_State
crfstate(const vector<Token> &vt, int i)
{
  CRF_State sample(vt[i].pos);

  string posm1 = "!BOS!", strm1 = "!BOS!"; // -1: previous position
  string pos0,  str0;                      //  0: current position
  string posp1 = "!EOS!", strp1 = "!EOS!"; // +1: next position

  string::size_type p = vt[i].str.find_last_of('/');
  str0 = vt[i].str.substr(0, p);
  pos0 = vt[i].str.substr(p+1);

  if (i >= 1) {
    string::size_type p = vt[i-1].str.find_last_of('/');
    strm1 = vt[i-1].str.substr(0, p);
    posm1 = vt[i-1].str.substr(p+1);
  }

  if (i < (int)vt.size() - 1) {
    string::size_type p = vt[i+1].str.find_last_of('/');
    strp1 = vt[i+1].str.substr(0, p);
    posp1 = vt[i+1].str.substr(p+1);
  }

  sample.add_feature("W0_" + str0);
  sample.add_feature("P0_" + pos0);

  sample.add_feature("W-1_" + strm1);
  sample.add_feature("P-1_" + posm1);

  sample.add_feature("W+1_" + strp1);
  sample.add_feature("P+1_" + posp1);

  //  cout << str0 << pos0 << endl;
  //  exit(0);

  return sample;
}
*/


void defaultFeatureGenerator(
    const Sentence &vt,
    int i,
	  CRF_State &sample )
{
  string str = vt[i].str;
  //  string str = normalize(vt[i].str);

  //sample.label = vt[i].pos;

  sample.add_feature("W0_" + vt[i].str);

  sample.add_feature("NW0_" + normalize(str));

  string prestr = "BOS";
  if (i > 0) prestr = vt[i-1].str;
  //  if (i > 0) prestr = normalize(vt[i-1].str);

  string prestr2 = "BOS";
  if (i > 1) prestr2 = vt[i-2].str;
  //  if (i > 1) prestr2 = normalize(vt[i-2].str);

  string poststr = "EOS";
  if (i < (int)vt.size()-1) poststr = vt[i+1].str;
  //  if (i < (int)vt.size()-1) poststr = normalize(vt[i+1].str);

  string poststr2 = "EOS";
  if (i < (int)vt.size()-2) poststr2 = vt[i+2].str;
  //  if (i < (int)vt.size()-2) poststr2 = normalize(vt[i+2].str);


  sample.add_feature("W-1_" + prestr);
  sample.add_feature("W+1_" + poststr);

  sample.add_feature("W-2_" + prestr2);
  sample.add_feature("W+2_" + poststr2);
#if 0
  sample.add_feature("W-10_" + prestr + "_" + str);
  sample.add_feature("W0+1_" + str  + "_" + poststr);
  sample.add_feature("W-1+1_" + prestr  + "_" + poststr);
#endif

  //sample.add_feature("W-10+1_" + prestr  + "_" + str + "_" + poststr);

  //  sample.add_feature("W-2-1_" + prestr2  + "_" + prestr);
  //  sample.add_feature("W+1+2_" + poststr  + "_" + poststr2);

  // train = 10000 no effect
  //  if (i > 0 && prestr.size() >= 3)
  //    sample.add_feature("W-1S_" + prestr.substr(prestr.size()-3));
  //  if (i < (int)vt.size()-1 && poststr.size() >= 3)
  //    sample.add_feature("W+1S_" + poststr.substr(poststr.size()-3));

  // sentence type
  //  sample.add_feature("ST_" + vt[vt.size()-1].str);

  for (size_t j = 1; j <= 10; j++) {
    char buf[1000];
    //    if (str.size() > j+1) {
    if (str.size() >= j) {
      sprintf(buf, "SUF%d_%s", (int)j, str.substr(str.size() - j).c_str());
      sample.add_feature(buf);
    }
    //    if (str.size() > j+1) {
    if (str.size() >= j) {
      sprintf(buf, "PRE%d_%s", (int)j, str.substr(0, j).c_str());
      sample.add_feature(buf);
    }
  }

  for (size_t j = 0; j < str.size(); j++) {
    if (isdigit(str[j])) {
      sample.add_feature("CTN_NUM");
      break;
    }
  }

  /*if (str.size() > 0 && isupper(str[0]))
      sample.add_feature("CTN_UPF");*/

  for (size_t j = 0; j < str.size(); j++) {
    if (isupper(str[j])) {
      sample.add_feature("CTN_UPP");
      break;
    }
  }
  for (size_t j = 0; j < str.size(); j++) {
    if (str[j] == '-') {
      sample.add_feature("CTN_HPN");
      break;
    }
  }
  bool allupper = true;
  for (size_t j = 0; j < str.size(); j++) {
    if (!isupper(str[j])) {
      allupper = false;
      break;
    }
  }
  if (allupper) sample.add_feature("ALL_UPP");

  if (WNdic.size() > 0) {
    const string n = normalize(str);
    for (map<string, string>::const_iterator i = WNdic.lower_bound(n); i != WNdic.upper_bound(n); i++) {
      sample.add_feature("WN_" + i->second);
    }
  }
  //  for (int j = 0; j < vt.size(); j++)
  //    cout << vt[j].str << " ";
  //  cout << endl;
  //  cout << i << endl;

  //  cout << sample.label << "\t";
  //  for (vector<string>::const_iterator j = sample.features.begin(); j != sample.features.end(); j++) {
  //      cout << *j << " ";
  //  }
  //  cout << endl;
}


Trainer::Trainer(
    FeatureCallback *callback ) : featureCallback(callback)
{
    if (featureCallback == NULL)
        featureCallback = defaultFeatureGenerator;
}


Trainer::~Trainer()
{

}


CRF_Model &Trainer::getModel()
{
    return model;
}


void Trainer::train(
    const std::vector<Sentence> & vs,
    double gaussian,
    const bool use_l1 )
{
  CALLED_THIS;
  //if (method != CRF_Model::BFGS && use_l1) { cerr << "error: L1 regularization is currently not supported in this mode. Please use other optimziation methods." << endl; exit(1); }

  for (vector<Sentence>::const_iterator i = vs.begin(); i != vs.end(); i++) {
    const Sentence & s = *i;
    CRF_Sequence cs;
    for (size_t j = 0; j < s.size(); j++)
    {
      CRF_State state;
      state.label = s[j].pos;
      featureCallback(s, j, state);
      cs.add_state(state);
    }
    model.add_training_sample(cs);
  }
  //  m.set_heldout(50, 0);

  if (use_l1)
      model.train(0, 0, 1.0);
  else
      model.train(0, gaussian);

  //  m.save_to_file("model.crf");
}



}}
