/*
 * $Id$
 */

#include <stdio.h>
#include <fstream>
#include <map>
#include <list>
#include <set>
#include <iostream>
#include <cfloat>
#include <sstream>
//#include "maxent.h"
#include <post/crf.hh>
#include <post/common.hh>

using namespace std;

string MODEL_DIR = "."; // the default directory for saving the models
extern int LOOKAHEAD_DEPTH;

//int eftrain(const vector<Sentence> & vs, const string & modeldir, const bool use_l1);
int
crftrain(const CRF_Model::OptimizationMethod method,
	 CRF_Model & c, const vector<Sentence> & vs, double g, const bool use_l1);


void read_tagged(istream * ifile, vector<Sentence> & vs)
{
  static ParenConverter paren_converter;
 
  string line;
  while (getline(*ifile,line)) {
    istringstream is(line);
    string s;
    Sentence sentence;
    while (is >> s) {
      string::size_type i = s.find_last_of('/');
      string str = s.substr(0, i);
      string pos = s.substr(i+1);
      //                cout << str << " ";

      //      string str0 = str;
      str = paren_converter.Ptb2Pos(str);
      //      if (str != str0) cout << str0 << " " << str << endl;
      //pos = paren_converter.Pos2Ptb(pos);

      //      cout << str << "\t" << pos << endl;
      Token t(str, pos);
      sentence.push_back(t);
    }
    vs.push_back(sentence);
    //    if (vs.size() >= num_sentences) break;
    //            cout << endl;
  }
  //  exit(0);
}

void print_help()
{
  cout << "Usage: lapos-learn [OPTION]... [FILE]" << endl;
  cout << "Create probabilistc models for the STePP tagger." << endl;
  cout << "The input must be one-sentence-per-line." << endl;
  cout << "The format for each line is: WORD1/POS1 WORD2/POS2 WORD3/POS3 ..." << endl;
  cout << endl;
  cout << "Mandatory arguments to long options are mandatory for short options too." << endl;
  cout << "  -m, --model=DIR       specify the directory for saving the models" << endl;
  cout << "  -c, --compact         build compact models" << endl;
  cout << "  -f, --fast            build only the model for the fast mode" << endl;
  cout << "  -h, --help            display this help and exit" << endl;
  cout << endl;
  cout << "With no FILE, or when FILE is -, read standard input." << endl;
  cout << endl;
  cout << "Report bugs to <tsuruoka@gmail.com>" << endl;
}

int main(int argc, char** argv)
{
  string ifilename;

  bool CRF_ONLY = false;
  CRF_Model::OptimizationMethod opmethod = CRF_Model::BFGS;

  double gaussian = 256;
  for (int i = 1; i < argc; i++) {
    string v = argv[i];
    //    if (v == "-i" && i < argc-1) ifilename = argv[i+1];
    //    if (v == "-o" && i < argc-1) ofilename = argv[i+1];
    if (v == "-d" && i < argc-1) {
      LOOKAHEAD_DEPTH = atoi(argv[i+1]);
      i++;
      continue;
    }
    if ( (v == "-m" || v == "--model") && i < argc-1) {
      MODEL_DIR = argv[i+1];
      i++;
      continue;
    }
    if (v.substr(0, 8) == "--model=") {
      MODEL_DIR = v.substr(8);
      continue;
    }
    if (v == "-h" || v == "--help") {
      print_help();
      exit(0);
    }
    if (v == "-") {
      ifilename = "";
      continue;
    }
    if (v[0] == '-') {
      cerr << "error: unknown option " << v << endl;
      cerr << "Try `stepp-learn --help' for more information." << endl;
      exit(1);
    }
    ifilename = v;
  }
  //  cerr << ifilename << endl;

  if (MODEL_DIR[MODEL_DIR.size()-1] != '/')  MODEL_DIR += "/";

  vector<Sentence> trains;

  istream *is(&std::cin);
  ifstream ifile;
  if (ifilename != "") {
    ifile.open(ifilename.c_str());
    if (!ifile) { cerr << "error: cannot open " << ifilename << endl; exit(1); }
    is = &ifile;
  }

  read_tagged(is, trains);

  if (trains.size() == 0) {
    if (ifilename != "") { 
      cerr << "error: cannot read \"" << ifilename << "\"" << endl; exit(1); 
    }
    cerr << "error: no training data." << endl;
    exit(1);
  }
  
  //  if (!CRF_ONLY) eftrain(trains, MODEL_DIR, use_l1_regularization);

  CRF_Model crfm;
  //      crfm.set_heldout(10000);
  crftrain(CRF_Model::PERCEPTRON, crfm, trains, 0, false);
  string mfile = MODEL_DIR + "model.la";
  //  cerr << "saving the CRF model to " << mfile << "...";
  crfm.save_to_file(mfile, 0.001);
  //  cerr << "done" << endl;

  cerr << endl;
  cerr << "the models have been saved in the directory \"" << MODEL_DIR << "\"." << endl;
}

/*
 * $Log$
 */

