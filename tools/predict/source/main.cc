#include <stdio.h>
#include <fstream>
#include <map>
#include <list>
#include <set>
#include <iomanip>
#include <iostream>
#include <cfloat>
#include <sstream>
#include <post/Predictor.hh>
#include <sys/time.h>

using namespace std;
using namespace nlptools::postagger;

bool PERFORM_TOKENIZATION = false;
bool OUTPUT_TAG_PROBS = false;
bool STANDOFF = false;
bool UIMA     = false;
bool ENJU     = false;


namespace nlptools {
namespace postagger {

void tokenize(const string & s, Sentence & vt, const bool use_upenn_tokenizer);

}}


ParenConverter paren_converter;

void main_usage()
{
		std::cerr << "Usage: tool_predict -m <model file> [ -i <input file> ]" << endl << endl;
		std::cerr << "Annotate each word in the input file with a part-of-speech tag." << endl << endl;
		std::cerr << "By default, the tagger assumes that input file contains one sentence per line" << endl;
		std::cerr << "and the words are tokenized with white spaces." << endl;
		//  cout << "Try -t and -s options if you want the tagger to process raw text." << endl;
		std::cerr << "Use -t option if you want to process untokenized sentences." << endl;
		std::cerr << endl;
		std::cerr << "Mandatory arguments to long options are mandatory for short options too." << endl;
		std::cerr << "  -m           Path to the model file" << endl;
		std::cerr << "  -t           Perform tokenization in each input sentence" << endl;
		std::cerr << "  -s,          Output in stand-off format" << endl;
		std::cerr << "  -u,          Output in UIMA format" << endl;
		std::cerr << "  -e,          Output in Enju format" << endl;
		std::cerr << "  -h, --help   Display this help and exit" << endl;
		std::cerr << endl;
		std::cerr << "If no input file name was given, the program read from standard input." << endl;
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

struct TagProb
{
	string tag;
	double prob;
	TagProb(const string & t_, const double & p_) : tag(t_), prob(p_) {}
	bool operator<(const TagProb & x) const { return prob > x.prob; }
};

int main(int argc, char** argv)
{
	string WORDNET_DIR = "";

	string modelFileName;
	string inputFileName;

	for (int i = 1; i < argc; i++)
	{
		string current = argv[i];
		string next;
		if (i + 1 < argc) next = argv[i + 1];

		if (current == "-m" && !next.empty())
		{
			modelFileName = next;
			i++;
			continue;
		}
		else
		if (current == "-i" && !next.empty())
		{
			inputFileName = next;
			i++;
			continue;
		}
		if (current == "-t")
		{
			PERFORM_TOKENIZATION = true;
			continue;
		}
		else
		if (current == "-s")
		{
			STANDOFF = true;
			continue;
		}
		else
		if (current == "-e")
		{
			ENJU = true;
			continue;
		}
		else
		if (current == "-u")
		{
			UIMA = true;
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

	istream *is(&std::cin);
	ifstream ifile;
	if (inputFileName != "")
	{
		ifile.open(inputFileName.c_str());
		if (!ifile) main_error("Can not open the input file '" + inputFileName + "'");
		is = &ifile;
	}

	Model crfm;
	if (crfm.load_from_file(modelFileName, false) == false)
		main_error("Can not load the model at '" + modelFileName + "'");

	string line;
	int nlines = 0;
	while (getline(*is, line))
	{
		nlines++;
		Sentence vt;
		tokenize(line, vt, PERFORM_TOKENIZATION);

		if (vt.size() > 990)
		{
			cerr << "warning: the sentence is too long. it has been truncated." << endl;
			while (vt.size() > 990) vt.pop_back();
		}

		// convert parantheses
		vector<string> org_strs;
		for (Sentence::iterator i = vt.begin(); i != vt.end(); i++)
		{
			org_strs.push_back(i->str);
			i->str = paren_converter.Ptb2Pos(i->str);
			i->prd = "?";
		}

		if (STANDOFF) cout << line << endl;
		if (vt.size() == 0)
		{
			cout << endl;
			continue;
		}

		// tag the words
		vector< map<string, double> > tagp;
		nlptools::postagger::Predictor predictor(crfm);
		predictor.predict(vt, tagp);

		// print the resutls
		for (size_t i = 0; i < vt.size(); i++)
		{
			const string s = org_strs[i];
			const string p = vt[i].prd;
			if (STANDOFF || OUTPUT_TAG_PROBS || UIMA || ENJU)
			{
				if (STANDOFF || UIMA || ENJU)
				{
					cout << vt[i].begin << "\t" << vt[i].end;
					if (!UIMA && !ENJU)
					{
						cout << "\t";
					}
				}
				if (!UIMA && !ENJU)
				{
					cout << s;
				}
				if (OUTPUT_TAG_PROBS)
				{
					vector<TagProb> tp;
					double sum = 0;
					for (map<string, double>::iterator j = tagp[i].begin(); j != tagp[i].end(); j++)
					{
						tp.push_back(TagProb(j->first, j->second));
						sum += j->second;
					}
					sort(tp.begin(), tp.end());
					for (vector<TagProb>::iterator j = tp.begin(); j != tp.end(); j++)
					{
						const double p = j->prob / sum; // normalize
						if (p == 1) cout << resetiosflags(ios::fixed);
						else        cout << setiosflags(ios::fixed) << setprecision(3);
						cout << "\t" << j->tag << "\t" << p;
					}
				}
				else
				{
					cout << "\t" + p;
					if (ENJU)
					{
						cout << "\t1";
					}
				}
				if (UIMA){
					cout << "\t0";
				}
				cout << endl;
			}
			else
			{
				if (i == 0)
					cout << s + "/" + p;
				else
					cout << " " + s + "/" + p;
			}
		}
		cout << endl;
	}
}
