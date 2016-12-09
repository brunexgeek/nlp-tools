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
#include <post/Trainer.hh>


using namespace std;


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
	cout << "Usage: tool_train -m <model> [ -i <training-set> ]" << endl;
	cout << endl;
	cout << "Create a model from the given training set." << endl;
	cout << endl;
	cout << "The input must be one-sentence-per-line." << endl;
	cout << "The format for each line is: WORD1/TAG1 WORD2/TAG2 WORD3/TAG3 ..." << endl;
	cout << endl;
	cout << "  -m          Path where the generated model wil be saved." << endl;
	cout << "  -i          Path to the training set." << endl;
	cout << "  -d          Specifies the lookahead depth (1-3). The default is 2." << endl;
	cout << "  -h          Display this help and exit" << endl;
	cout << endl;
	cout << "If no training set file name was given, the program read from standard input." << endl;
	cout << endl;
	cout << "To report bugs, open an issue at <https://github.com/brunexgeek/nlp-tools>" << endl;
}

int main(int argc, char** argv)
{
	string samplesFileName;
	string modelFileName;
	int lookaheadDepth = 2;

	for (int i = 1; i < argc; i++)
	{
		string current = argv[i];
		string next;
		if (i + 1 < argc) next = argv[i + 1];

		if (current == "-d" && !next.empty())
		{
			lookaheadDepth = atoi(next.c_str());
			if (lookaheadDepth < 1)
				lookaheadDepth = 1;
			else
			if (lookaheadDepth > 3)
				lookaheadDepth = 3;
			i++;
		}
		else
		if (current == "-m" && !next.empty())
		{
			modelFileName = next;
			i++;
			continue;
		}
		else
		if (current == "-i" && !next.empty())
		{
			samplesFileName = next;
			i++;
			continue;
		}
		else
		if (current == "-h" || current == "--help")
		{
			print_help();
			exit(0);
		}
		else
		{
			cerr << "error: unknown option " << current << endl;
			cerr << "Try `stepp-learn --help' for more information." << endl;
			exit(1);
		}
	}

	if (modelFileName.empty())
	{
		print_help();
		exit(0);
	}

	vector<Sentence> trains;

	istream *is(&std::cin);
	ifstream ifile;
	if (!samplesFileName.empty())
	{
		ifile.open(samplesFileName.c_str());
		if (!ifile)
		{
			cerr << "error: cannot open " << samplesFileName << endl;
			exit(1);
		}
		is = &ifile;
	}

	read_tagged(is, trains);

	if (trains.size() == 0)
	{
		if (!samplesFileName.empty())
		{
			cerr << "error: cannot read \"" << samplesFileName << "\"" << endl; exit(1);
		}
		cerr << "error: no training data." << endl;
		exit(1);
	}

	//  if (!CRF_ONLY) eftrain(trains, MODEL_DIR, use_l1_regularization);

	nlptools::postagger::Trainer trainer;
	//CRF_Model crfm;
	//      crfm.set_heldout(10000);
	trainer.train(/*CRF_Model::PERCEPTRON, crfm,*/ trains, 0, false);
	cerr << "Saving model to '" << modelFileName << "' ...";
	trainer.getModel().save_to_file(modelFileName, 0.001);
	cerr << "done" << endl;
}

/*
 * $Log$
 */

