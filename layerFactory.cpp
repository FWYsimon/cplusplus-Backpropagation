#include "layerFactory.hpp"

LayerFactory* LayerFactory::this_instance_ = nullptr;
LayerFactory* LayerFactory::getInstance(){
	if (!this_instance_){
		this_instance_ = new LayerFactory();
	}
	return this_instance_;
}

void LayerFactory::releaseInstance(){
	if (this_instance_){
		delete this_instance_;
		this_instance_ = nullptr;
	}
}

Layer* LayerFactory::getLayer(const char* type){
	if (all_instance_.find(type) != all_instance_.end())
		return all_instance_[type];
	return nullptr;
}

#define DefLayer(cls)	all_instance_[#cls] = new cls();
LayerFactory::LayerFactory(){
	DefLayer(DataLayer);
	DefLayer(InnerProductLayer);
}

LayerFactory::~LayerFactory(){
	for (auto& item : all_instance_)
		delete item.second;
	all_instance_.clear();
}