#include <post/Corpus.hh>
#include <fstream>
#include <string>
#include <sstream>


namespace nlptools {
namespace postagger {


using std::ifstream;
using std::string;
using std::istringstream;



Corpus::Corpus()
{

}


Corpus::~Corpus()
{

}


size_t Corpus::load(
	const std::string &fileName,
	char separator )
{
	ifstream input(fileName.c_str());
	if (input.good())
	{
		load(input, separator);
		input.close();
	}

	return sentences.size();
}


size_t Corpus::load(
	std::istream &input,
	char separator )
{
	static ParenConverter paren_converter;

	string line;
	while (getline(input,line))
	{
		istringstream is(line);
		string s;
		Sentence sentence;
		while (is >> s)
		{
			string str, pos;

			string::size_type i = s.find_last_of(separator);
			if (i != string::npos)
			{
				str = s.substr(0, i);
				pos = s.substr(i+1);
			}
			else
			{
				str = s;
			}

			//      string str0 = str;
			str = paren_converter.Ptb2Pos(str);
			//      if (str != str0) cout << str0 << " " << str << endl;
			//      pos = paren_converter.Pos2Ptb(pos);

			//      cout << str << "\t" << pos << endl;
			Token t(str, pos);
			sentence.push_back(t);
		}
		sentences.push_back(sentence);
		//if (vs.size() >= num_sentences) break;
	}

	return sentences.size();
}


void Corpus::clear()
{
	sentences.clear();
}


size_t Corpus::size() const
{
	return sentences.size();
}


Corpus::operator const std::vector<Sentence>&() const
{
	return sentences;
}


}}
