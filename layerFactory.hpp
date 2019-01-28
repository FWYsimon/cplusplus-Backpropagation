#include <map>
#include <string>
#include "layer.hpp"

using namespace std;

class LayerFactory {
public:
	LayerFactory();
	virtual ~LayerFactory();
	Layer* getLayer(const char* type);
	static LayerFactory* getInstance();
	static void releaseInstance();

private:
	static LayerFactory* this_instance_;
	map<string, Layer*> all_instance_;
};