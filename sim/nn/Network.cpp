#include "Network.hpp"

Node::Node(Layer* p, bool usesig) : usesig(usesig) {
	if (!p) return;
	parents = std::vector<Node*>(p->nodes);
	size = parents.size();
	weights.resize(size);
	for (uint a = 0; a < size; a++) weights[a] = (rand() % 100)*0.1f - 5;
	bias = (rand() % 100)*0.1f - 5;
}

void Node::Calc() {
	value = output = 0;
	for (uint a = 0; a < size; a++) {
		value += parents[a]->output*weights[a];
	}
	value += bias;
	output = usesig? sigmoid(value) : value;
}

void Node::Cost(double tar, double _a) {
	double res = 0.5f * pow(output - tar, 2);
	double res2 = 0.5f * pow((usesig? sigmoid(value + _a) : value + _a) - tar, 2);

	cost += res;
	double v2 = usesig ? isigmoid(output + _a) : output + _a;
	doutput += (res2 - res);
}

void Node::BP(double _a) {
	bias -= doutput * _a;
	for (uint a = 0; a < size; a++) {
		weights[a] -= doutput * _a * parents[a]->output;
		parents[a]->doutput += doutput * _a * weights[a];
	}
}


Layer::Layer(uint sz, Layer* p, bool usesig) {
	size = sz;
	nodes.resize(sz);
	for (uint a = 0; a < size; a++) {
		nodes[a] = new Node(p, usesig);
	}
}

void Layer::Set(const double* vals) {
	for (uint a = 0; a < size; a++) {
		nodes[a]->output = vals[a];
	}
}

void Layer::Calc() {
	for (uint a = 0; a < size; a++) {
		nodes[a]->Calc();
	}
}

void Layer::Clc() {
	for (uint a = 0; a < size; a++) {
		nodes[a]->doutput = 0;
		nodes[a]->cost = 0;
	}
}

void Layer::Cost(const double* tars, double _a) {
	for (uint a = 0; a < size; a++) {
		nodes[a]->Cost(tars[a], _a);
	}
}

void Layer::BP(const double _a) {
	for (uint a = 0; a < size; a++) {
		nodes[a]->BP(_a);
	}
}


Net::Net(uint ls, uint* ns): me(this), size(ls) {
	layers.resize(ls);
	layers[0] = new Layer(ns[0]);
	for (uint a = 1; a < size; a++) {
		layers[a] = new Layer(ns[a], layers[a-1], a != (size-1));
	}
}

void Net::Eval(const double* vals) {
	layers[0]->Set(vals);
	for (uint a = 1; a < size; a++) {
		layers[a]->Calc();
	}
}

void Net::BP(uint cnt, const std::vector<std::pair<std::vector<double>, double>>& set, double _a) {
	for (uint a = 0; a < size; a++)
		layers[a]->Clc();
	for (uint a = 0; a < cnt; a++) {
		Eval(&set[a].first[0]);
		layers[size - 1]->Cost(&set[a].second, _a); //we have only 1 target anyway
	}
	for (uint a = size - 1; a > 1; a--) {
		layers[a]->BP(_a);
	}
}