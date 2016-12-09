#ifndef NLPTOOLS_POST_TRAINER_HH
#define NLPTOOLS_POST_TRAINER_HH


#include <post/crf.hh>
#include <post/common.hh>
#include <vector>

namespace nlptools {
namespace postagger {


class Trainer
{
	public:
		Trainer(
			FeatureCallback *featureCallback = NULL );

		~Trainer();

		void train(
			const std::vector<Sentence> & vs,
			double gaussian,
			const bool use_l1 = false );

		CRF_Model &getModel();

	private:
		FeatureCallback *featureCallback;
		CRF_Model model;
};


}}



#endif // NLPTOOLS_POST_TRAINER_HH