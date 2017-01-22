#ifndef NLPTOOLS_POST_CORPUS_HH
#define NLPTOOLS_POST_CORPUS_HH


#include <post/Model.hh>
#include <post/common.hh>
#include <vector>
#include <string>
#include <iostream>


namespace nlptools {
namespace postagger {


class Corpus
{
	public:
		Corpus();

		~Corpus();

		size_t load(
			const std::string &fileName,
			char separator = '/' );

		size_t load(
			std::istream &input,
			char separator = '/' );

		size_t size() const;

		void clear();

		operator const std::vector<Sentence>&() const;

	private:
		std::vector<Sentence> sentences;
};


}}

#endif // NLPTOOLS_POST_CORPUS_HH