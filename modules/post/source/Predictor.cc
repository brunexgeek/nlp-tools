#include <post/Predictor.hh>
#include <map>
#include <string>



namespace nlptools {
namespace postagger {


void defaultFeatureGenerator(
	const Sentence &vt,
	int i,
	CRF_State &sample );


Predictor::Predictor(
	Model &model,
	FeatureCallback *callback ) : featureCallback(callback),
		model(model)
{
	if (callback == NULL)
		featureCallback = defaultFeatureGenerator;
}


Predictor::~Predictor()
{

}


Model &Predictor::getModel()
{
	return model;
}


void Predictor::predict(
	Sentence & s,
	std::vector< std::map<std::string, double> > & tagp )
{
	CRF_Sequence cs;
	for (size_t j = 0; j < s.size(); j++)
	{
		CRF_State state;
		featureCallback(s, (int)j, state);
		cs.add_state(state);
	}

	model.decode_lookahead(cs);

	tagp.clear();
	for (size_t k = 0; k < s.size(); k++)
	{
		s[k].prd = cs[k].label;
		std::map<std::string, double> vp;
		vp[s[k].prd] = 1.0;
		tagp.push_back(vp);
	}
}


}}