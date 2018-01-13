#include <iostream>
#include <fstream>
#include <time.h>
#include <random>

#include "Network.h"

std::vector<std::pair<std::vector<double>, double>> dataset;
uint datasetSz;

void ReadDataFile() {
	std::ifstream strm("D:\\data.txt", std::ios::binary);
	double f;
	while (1) {
		strm >> f;
		if (strm.eof()) return;
		std::pair<std::vector<double>, double> data;
		data.first.resize(10, f);
		for (uint a = 1; a < 10; a++) {
			strm >> data.first[a];
			if (strm.eof()) return;
		}
		strm >> f;
		data.second = (log10(f)/2 - 2)/2.5;
		if (strm.eof()) return;

		dataset.push_back(data);
		datasetSz++;
	}
}

Net* net;

int main() {
	srand(time(NULL));

	ReadDataFile();

	uint szs[4] = { 10, 20, 10, 1 };
	net = new Net(4, szs);

	while (1) {
		for (uint a = 0; a < 500; a++) {
			net->BP(datasetSz, dataset, 0.01f);
		}
		std::cout << net->layers[3]->nodes[0]->cost << std::endl;
	}

	return 0;
}