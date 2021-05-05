#include <ctime>
#include <iostream>

#include "Clustering.hpp"
#include "Distances.hpp"
#include "Tools.hpp"

const static std::vector<int> R_PARAMETERS = {1, 2, 4, 8, 16, 32, 64, 128};

std::tuple<float, float, float>
runKNN(const std::pair<std::vector<int>, Eigen::SparseMatrix<int>>& trainDataset,
       const std::pair<std::vector<int>, Eigen::SparseMatrix<int>>& testDataset,
       const Eigen::MatrixXf& embeddings,
       int r,
       int k,
       const std::string& functionName,
       bool verbose) {
  std::vector<std::unordered_map<int, int>> testHashes, trainHashes;

  if(functionName == "rel-rwmd") {
    testHashes.resize(testDataset.first.size());
    for (int i = 0; i < testHashes.size(); i++) {
      Eigen::SparseVector<int> doc = testDataset.second.col(i);
      testHashes[i].reserve(doc.nonZeros());
      for (Eigen::SparseVector<int>::InnerIterator it(doc); it; ++it) {
	testHashes[i][it.row()] = it.value();
      }
    }
    
    trainHashes.resize(trainDataset.first.size());
    for (int i = 0; i < trainHashes.size(); i++) {
      Eigen::SparseVector<int> doc = trainDataset.second.col(i);
      trainHashes[i].reserve(doc.nonZeros());
      for (Eigen::SparseVector<int>::InnerIterator it(doc); it; ++it) {
	trainHashes[i][it.row()] = it.value();
      }
    }
  }

  std::time_t preStart = std::clock();
  std::pair<std::vector<std::unordered_map<int, int>>, int> relatedWordsCache;
  if(functionName == "rel-wmd" ||
     functionName == "rel-rwmd" ||
     functionName == "lc-rel-rwmd") {
    relatedWordsCache
      = Tools::computeRelatedWordsCacheFromEmbeddings(embeddings,
						      r);
  }

  Eigen::MatrixXi trainCentroids, trainNearDistancesToDocs;
  std::vector<std::unordered_map<int, int>> trainNearRelatedDistancesToDocs;
  if(functionName == "wcd") {
    trainCentroids
      = Distances::computeWCD(trainDataset.second,
			      embeddings);
  } else if(functionName == "lc-rwmd") {
    trainNearDistancesToDocs
      = Tools::computeNearestDistancesToDocs(trainDataset.second,
					     embeddings);
  } else if(functionName == "lc-rel-rwmd") {
    trainNearRelatedDistancesToDocs
      = Tools::computeNearestRelatedDistancesToDocs(trainDataset.second,
						    relatedWordsCache.first);
  }
  
  float preTime = (std::clock() - preStart) / (float) CLOCKS_PER_SEC;

  int numLabels = 1 + (*std::max_element(trainDataset.first.begin(), trainDataset.first.end()));

  std::time_t knnStart = std::clock();
  std::vector<int> predictedLabels(testDataset.first.size());
  std::vector<int> predictedLabels2(testDataset.first.size());
  std::vector<std::pair<float, int>> distancePairs(trainDataset.first.size());
  std::vector<std::pair<float, int>> distancePairs2(trainDataset.first.size());
  Eigen::SparseVector<int> doc1, doc2;
  Eigen::MatrixXi doc1Cache;
  Eigen::VectorXi centroid1, centroid2, nearDistancesToDoc1;
  std::unordered_map<int, int> nearRelatedDistancesToDoc1;
  std::unordered_map<int, int> doc1Hash;
  for(int i = 0; i < testDataset.first.size(); i++) {
    doc1 = testDataset.second.col(i);

    if(functionName == "wcd") {
      centroid1 = Distances::computeWCD(doc1, embeddings);
    } else if(functionName == "rel-rwmd") {
      doc1Hash = testHashes[i];
    } else if(functionName == "lc-rwmd") {
      nearDistancesToDoc1
	= Tools::computeNearestDistancesToDoc(doc1, embeddings);
    } else if(functionName == "lc-rel-rwmd") {
      nearRelatedDistancesToDoc1
	= Tools::computeNearestRelatedDistancesToDoc(doc1,
						     relatedWordsCache.first);
    }

    for(int j = 0; j < trainDataset.first.size(); j++) {
      if(functionName == "wcd") {
	centroid2 = trainCentroids.col(j);
      } else {
	doc2 = trainDataset.second.col(j);
      }

      std::time_t funcStart = std::clock();
      if(functionName == "cosine") {
	distancePairs[j].first
	  = Distances::computeCosineDistance(doc1, doc2);
      } else if(functionName == "wcd") {
	distancePairs[j].first
	  = Distances::computeEuclideanDistance(centroid1, centroid2);
      } else if(functionName == "wmd") {
	distancePairs[j].first
	  = Distances::computeWMD(doc1, doc2, embeddings);
      } else if(functionName == "rwmd") {
	distancePairs[j].first
	  = Distances::computeRWMD(doc1, doc2, embeddings);
      } else if(functionName == "lc-rwmd") {
        distancePairs[j].first
	  = Distances::computeLinearRWMD(doc1,
					 nearDistancesToDoc1,
					 doc2,
					 trainNearDistancesToDocs.col(j));
      } else if(functionName == "rel-wmd") {
	distancePairs[j].first
	  = Distances::computeRelWMD(doc1,
				     doc2,
				     r,
				     relatedWordsCache);
      } else if(functionName == "rel-rwmd") {
	distancePairs[j].first
	  = Distances::computeRelRWMD(doc1,
				      doc1Hash,
				      doc2,
				      trainHashes[j],
				      relatedWordsCache);
      } else if(functionName == "lc-rel-rwmd") {
	distancePairs[j].first
	  = Distances::computeLinearRelatedRWMD(doc1,
						nearRelatedDistancesToDoc1,
						doc2,
						trainNearRelatedDistancesToDocs[j],
						relatedWordsCache.second);
      } else {
	throw std::invalid_argument("Unknown function: " + functionName);
            distancePairs[j].second = trainDataset.first[j];
      }
      
      float funcTime = (std::clock() - funcStart) / (float) CLOCKS_PER_SEC;
      if(verbose) {
	std::cout << doc1.nonZeros() << "\t"
		  << doc2.nonZeros() << "\t"
		  << funcTime << "\t"
		  << distancePairs[j].first
		  << std::endl;
      }
    }
    predictedLabels[i] = Tools::computePredictedLabel(distancePairs, numLabels, k);
  }
  float knnTime = (std::clock() - knnStart) / (float) CLOCKS_PER_SEC;

  float errorRate = Tools::computeErrorRate(predictedLabels,
					    testDataset.first);

  return std::make_tuple(errorRate, preTime, knnTime);
}

int
selectBestRForKusner(const std::pair<std::vector<int>, Eigen::SparseMatrix<int>>& dataset,
		     const Eigen::MatrixXf& embeddings,
		     int k,
		     const std::string& functionName,
		     bool verbose)
{

  if(functionName != "rel-wmd" &&
     functionName != "rel-rwmd" &&
     functionName != "lc-rel-rwmd") {
    return -1;
  }

  std::size_t numDocs = dataset.first.size();
  std::vector<int> indices(numDocs, 0);
  for(int i = 0; i < numDocs; i++) {
    indices[i] = i;
  }

  std::random_shuffle(indices.begin(), indices.end());

  std::size_t numPartitions = 5;
  std::size_t numDocsPerPartition
    = (std::size_t) std::ceil((numDocs + 1.0f) / numPartitions);

  std::vector<std::vector<int>> partitions(numPartitions);
  for(int i = 0; i < numPartitions; i++) {
    partitions[i].reserve(numDocsPerPartition);
  }

  for(int i = 0; i < numDocs; i++) {
    std::size_t idx = i / numDocsPerPartition;
    partitions[idx].push_back(indices[i]);
  }

  std::vector<float> testErrors(R_PARAMETERS.size(), 0.0f);
  for(std::size_t i = 0; i < numPartitions; i++) {
    std::size_t testSize = partitions[i].size();
    std::size_t trainSize = numDocs - testSize;

    std::vector<int> trainLabels(trainSize, -1);
    std::vector<int> testLabels(testSize, -1);
    std::vector<Eigen::Triplet<int>> trainTriplets;
    std::vector<Eigen::Triplet<int>> testTriplets;

    std::size_t trainIdx = 0, testIdx = 0;
    for(std::size_t j = 0; j < numPartitions; j++) {
      if(i == j) {
	for(int docIdx: partitions[j]) {
	  for (Eigen::SparseMatrix<int>::InnerIterator it(dataset.second, docIdx); it; ++it) {
	    testTriplets.emplace_back(it.row(), testIdx, it.value());
	  }
	  testLabels[testIdx] = dataset.first[docIdx];
	  testIdx++;
	}
      } else {
	for(int docIdx: partitions[j]) {
	  for (Eigen::SparseMatrix<int>::InnerIterator it(dataset.second, docIdx); it; ++it) {
	    trainTriplets.emplace_back(it.row(), trainIdx, it.value());
	  }
	  trainLabels[trainIdx] = dataset.first[docIdx];
	  trainIdx++;
	}
      }
    }

    Eigen::SparseMatrix<int> trainDocuments(dataset.second.rows(),
					    trainSize);
    Eigen::SparseMatrix<int> testDocuments(dataset.second.rows(),
					   testSize);
    trainDocuments.setFromTriplets(trainTriplets.begin(),
				   trainTriplets.end());
    testDocuments.setFromTriplets(testTriplets.begin(),
				  testTriplets.end());

    std::pair<std::vector<int>, Eigen::SparseMatrix<int>> trainDataset
      = std::make_pair(trainLabels, trainDocuments);
    std::pair<std::vector<int>, Eigen::SparseMatrix<int>> testDataset
      = std::make_pair(testLabels, testDocuments);
    for(std::size_t j = 0; j < R_PARAMETERS.size(); j++) {
      int r = R_PARAMETERS[j];
      std::tuple<float, float, float> result
	= runKNN(trainDataset,
		 testDataset,
		 embeddings,
		 r,
		 k,
		 functionName,
		 verbose);
      testErrors[j] += std::get<0>(result);
      std::cout << i << "\t"
		<< r << "\t"
		<< std::get<0>(result) << "\t"
		<<  std::get<1>(result) << "\t"
		<<  std::get<2>(result)
		<< std::endl;
    }
  }

  float minErrorRate = *std::min_element(testErrors.begin(),
					 testErrors.end());
  for(std::size_t j = 0; j < R_PARAMETERS.size(); j++) {
    if(testErrors[j] < 1.01 * minErrorRate) {
      return R_PARAMETERS[j];
    }
  }
  return -1;
}

void
runKusnerExperiment(const std::string& trainDatasetFilePath,
		    const std::string& testDatasetFilePath,
		    const std::string& embeddingsFilePath,
		    int k,
		    const std::string& functionName,
		    int r,
		    bool verbose)
{
  Eigen::MatrixXf embeddings = Tools::getEmbeddings(embeddingsFilePath);
  std::pair<std::vector<int>, Eigen::SparseMatrix<int>> trainDataset
    = Tools::getDocuments(trainDatasetFilePath, embeddings.cols());
  std::pair<std::vector<int>, Eigen::SparseMatrix<int>> testDataset
    = Tools::getDocuments(testDatasetFilePath, embeddings.cols());

  if(r < 1) {
    r = selectBestRForKusner(trainDataset,
			     embeddings,
			     k,
			     functionName,
			     verbose);
  }

  std::tuple<float, float, float> result = runKNN(trainDataset,
						  testDataset,
						  embeddings,
						  r,
						  k,
						  functionName,
						  verbose);
  float errorRate = std::get<0>(result);
  float preTime = std::get<1>(result);
  float knnTime = std::get<2>(result);

  // Dump results to Console
  std::string SEP = "\t";
  std::string filesString
    = trainDatasetFilePath + SEP
    + testDatasetFilePath + SEP
    + embeddingsFilePath + SEP;
  std::string paramsString
    = std::to_string(k) + SEP
    + functionName + SEP
    + std::to_string(r) + SEP;
  std::string timesString
    = std::to_string(preTime) + SEP
    + std::to_string(knnTime) + SEP;
  std::cout << filesString
	    << paramsString
	    << timesString
	    << errorRate
	    << std::endl;
}

/* ---------- */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int
main(int argc, const char * argv[])
{
  std::string trainDatasetFilePath;
  std::string testDatasetFilePath;
  std::string embeddingsFilePath;
  std::string functionName;
  int k = -1;
  int r = -1;
  bool verbose = false;

  char *mode = "kusner";

  // Parse cmd parameters
  for(int i = 1; i < argc; i+=2) {
    const char *cmd = argv[i];
    //printf("argv[%d]: %s\t%s\n", i, argv[i], argv[i+1]);
    if(strcmp(cmd, "--tr") == 0) {
      trainDatasetFilePath = argv[i+1];
    } else if(strcmp(cmd, "--te") == 0) {
      testDatasetFilePath = argv[i+1];
    } else if(strcmp(cmd, "--emb") == 0) {
      embeddingsFilePath = argv[i+1];
    } else if(strcmp(cmd, "--func") == 0) {
      functionName = argv[i+1];
    } else if(strcmp(cmd, "--k") == 0) {
      k = std::stoi(argv[i+1]);
    } else if(strcmp(cmd, "--r") == 0) {
      r = std::stoi(argv[i+1]);
    } else if(strcmp(cmd, "--verbose") == 0) {
      if (strcmp(argv[i+1], "true") == 0) verbose = true;
    } else if(strcmp(cmd, "--help") == 0) {
      printf("Usage: kusner [options]\n\n");
      printf("Input Options:\n\n");
      printf("--tr <filepath>:   %s\n\n", "Train dataset filepath");
      printf("--te <filepath>:   %s\n\n", "Test dataset filepath");
      printf("--emb <filepath>:  %s\n\n", "Embeddings filepath");
      printf("--func <function>: %s\n\n", "Function to be used");
      printf("--k X:             %s\n\n", "Number of neighbours in kNN");
      printf("--r X:             %s\n\n", "Number of related words");
      exit(0);
    } else {
      printf("Unknown option: %s\n", cmd);
      exit(1);
    }      
  }

  runKusnerExperiment(trainDatasetFilePath,
		      testDatasetFilePath,
		      embeddingsFilePath,
		      k,
		      functionName,
		      r,
		      verbose);
  
  exit(0);
}
