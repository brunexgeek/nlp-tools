#ifndef NLPTOOLS_POST_PREDICTOR_HH
#define NLPTOOLS_POST_PREDICTOR_HH


#include <post/Model.hh>
#include <post/common.hh>
#include <vector>

namespace nlptools {
namespace postagger {


class Predictor
{
	public:
		Predictor(
			Model &model,
			FeatureCallback *featureCallback = NULL );

		~Predictor();

		Model &getModel();

		void predict(
			Sentence & s,
			std::vector< std::map<std::string, double> > & tagp );

	private:
		FeatureCallback *featureCallback;
		Model &model;
};


}}



#endif // NLPTOOLS_POST_PREDICTOR_HH