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
#include <post/Corpus.hh>


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

void main_usage()
{
	std::cerr << "Usage: tool_train -m <model> [ -i <training-set> ]" << endl;
	std::cerr << endl;
	std::cerr << "Create a model from the given training set." << endl;
	std::cerr << endl;
	std::cerr << "The input must be one-sentence-per-line." << endl;
	std::cerr << "The format for each line is: WORD1/TAG1 WORD2/TAG2 WORD3/TAG3 ..." << endl;
	std::cerr << endl;
	std::cerr << "  -m          Path where the generated model wil be saved." << endl;
	std::cerr << "  -i          Path to the training set." << endl;
	std::cerr << "  -d          Specifies the lookahead depth (1-3). The default is 2." << endl;
	std::cerr << "  -h          Display this help and exit" << endl;
	std::cerr << endl;
	std::cerr << "If no training set file name was given, the program read from standard input." << endl;
	std::cerr << endl;
	std::cerr << "To report bugs, open an issue at <https://github.com/brunexgeek/nlp-tools>" << endl;
}

void main_error(
	const std::string &message )
{
	std::cerr << "Error: " << message << std::endl << std::endl;
	main_usage();
	exit(1);
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
			main_usage();
			exit(0);
		}
		else
		{
			main_error( "unknown option '" + current + "'");
			exit(1);
		}
	}

	if (modelFileName.empty()) main_error("missing model file name");

	nlptools::postagger::Corpus corpus;
	if (!samplesFileName.empty())
		corpus.load(samplesFileName);
	else
		corpus.load(std::cin);

	if (corpus.size() == 0)
	{
		if (!samplesFileName.empty())
			main_error("cannot read '" + samplesFileName + "'");
		else
			main_error("no training data.");
	}

	nlptools::postagger::Trainer trainer;
	trainer.train(corpus, 0, false);

	cerr << "Saving model to '" << modelFileName << "' ...";
	trainer.getModel().save_to_file(modelFileName, 0.001);
	cerr << "done" << endl;
}

