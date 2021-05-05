#include <ctime>
#include <iostream>

#include "Clustering.hpp"
#include "Distances.hpp"
#include "Tools.hpp"

std::tuple<float, float, float>
runRelatedDocumentPairsIdentification(const std::vector<std::tuple<int, int, int>>& tripletsDataset,
				      const Eigen::SparseMatrix<int>& documents,
				      const Eigen::MatrixXf& embeddings,
				      int numClusters,
				      int maxIterations,
				      const std::string& functionName,
				      int r,
				      bool verbose)
{
  std::vector<std::unordered_map<int, int>> docHashes;

  if(functionName == "rel-rwmd") {
    docHashes.resize(documents.cols());
    for(int i = 0; i < docHashes.size(); i++) {
      Eigen::SparseVector<int> doc = documents.col(i);
      docHashes[i].reserve(doc.nonZeros());
      for (Eigen::SparseVector<int>::InnerIterator it(doc); it; ++it) {
	docHashes[i][it.row()] = it.value();
      }
    }
  }

  std::time_t preStart = std::clock();
  std::pair<std::vector<std::unordered_map<int, int>>, int> relatedWordsCache;
  Eigen::MatrixXi centroids, nearDistancesToDocs;
  std::vector<std::unordered_map<int, int>> nearRelatedDistancesToDocs;
  if(functionName == "rel-wmd" ||
     functionName == "rel-rwmd" ||
     functionName == "lc-rel-rwmd") {
    if(numClusters > 0) {
      if (verbose) {
	std::cout << "Computing related words given clusters..." << std::endl;
      }
      std::pair<Eigen::MatrixXf, std::vector<int>> result
	= Clustering::computeWeightedKMeans(embeddings,
					    std::vector<float>(embeddings.cols(), 1.0f),
					    numClusters,
					    maxIterations);
      std::vector<std::vector<int>> embeddingClusters(numClusters,
						      std::vector<int>());
      for(int i = 0; i < result.second.size(); i++) {
	embeddingClusters[result.second[i]].push_back(i);
      }
      relatedWordsCache
	= Tools::computeRelatedWordsCacheFromClusters(embeddingClusters,
						      embeddings,
						      r,
						      verbose);
    } else {
      if(verbose) {
	std::cout << "Computing related words..." << std::endl;
      }
      relatedWordsCache
	= Tools::computeRelatedWordsCacheFromEmbeddings(embeddings,
							r,
							verbose);
    }

    if(functionName == "lc-rel-rwmd") {
      if(verbose) {
	std::cout << "Computing nearest related distances to each doc..."
		  << std::endl;
      }
      nearRelatedDistancesToDocs
	= Tools::computeNearestRelatedDistancesToDocs(documents,
						      relatedWordsCache.first);
    }
  } else if(functionName == "wcd") {
    centroids = Distances::computeWCD(documents, embeddings);
  } else if(functionName == "lc-rwmd") {
    nearDistancesToDocs
      = Tools::computeNearestDistancesToDocs(documents,
					     embeddings);
  }

  float preTime = (std::clock() - preStart) / (float) CLOCKS_PER_SEC;
  if(verbose) {
    std::cout << preTime << std::endl;
  }

  // Run experiment
  int numCorrect = 0;
  std::time_t experimentStart = std::clock();
  Eigen::SparseVector<int> doc1;
  Eigen::SparseVector<int> doc2;
  Eigen::SparseVector<int> doc3;
  Eigen::SparseVector<int> otherDoc;
  Eigen::VectorXi centroid1;
  Eigen::VectorXi centroid2;
  Eigen::VectorXi centroid3;
  Eigen::VectorXi otherCentroid;
  Eigen::VectorXi nearDistancesToDoc1;
  std::unordered_map<int, int> nearRelatedDistancesToDoc1;
  for(const std::tuple<int, int, int>& triplet: tripletsDataset) {
    int idx1 = std::get<0>(triplet);
    int idx2 = std::get<1>(triplet);
    int idx3 = std::get<2>(triplet);
    int otherIdx;
    if(functionName == "wcd") {
      centroid1 = centroids.col(idx1);
      centroid2 = centroids.col(idx2);
      centroid3 = centroids.col(idx3);
    } else {
      doc1 = documents.col(idx1);
      doc2 = documents.col(idx2);
      doc3 = documents.col(idx3);
      if(functionName == "lc-rwmd") {
	nearDistancesToDoc1 = nearDistancesToDocs.col(idx1);
      } else if(functionName == "lc-rel-rwmd") {
	nearRelatedDistancesToDoc1 = nearRelatedDistancesToDocs[idx1];
      }
    }

    if(functionName != "wcd" &&
       (doc1.nonZeros() == 0 ||
	doc2.nonZeros() == 0 ||
	doc3.nonZeros() == 0)) {
      continue;
    }

    float tripletDistances[2];
    for(int j = 0; j < 2; j++) {
      otherIdx = j == 0 ? idx2 : idx3;
      if(functionName == "wcd") {
	otherCentroid = j == 0? centroid2 : centroid3;
      } else {
	otherDoc = j == 0? doc2 : doc3;
      }

      std::time_t funcStart = std::clock();
      if(functionName == "cosine") {
	tripletDistances[j]
	  = Distances::computeCosineDistance(doc1,
					     otherDoc);
      } else if(functionName == "wcd") {
	tripletDistances[j]
	  = Distances::computeEuclideanDistance(centroid1,
						otherCentroid);
      }	else if(functionName == "wmd") {
	tripletDistances[j]
	  = Distances::computeWMD(doc1,
				  otherDoc,
				  embeddings);
      }	else if(functionName == "rwmd") {
	tripletDistances[j]
	  = Distances::computeRWMD(doc1,
				   otherDoc,
				   embeddings);
      } else if(functionName == "lc-rwmd") {
	tripletDistances[j]
	  = Distances::computeLinearRWMD(doc1,
					 nearDistancesToDoc1,
					 otherDoc,
					 nearDistancesToDocs.col(otherIdx));
      } else if(functionName == "rel-wmd") {
	tripletDistances[j]
	  = Distances::computeRelWMD(doc1,
				     otherDoc,
				     r,
				     relatedWordsCache);
      }	else if(functionName == "rel-rwmd") {
	tripletDistances[j]
	  = Distances::computeRelRWMD(doc1,
				      docHashes[idx1],
				      otherDoc,
				      docHashes[otherIdx],
				      relatedWordsCache);
      }	else if(functionName == "lc-rel-rwmd") {
	tripletDistances[j]
	  = Distances::computeLinearRelatedRWMD(doc1,
						nearRelatedDistancesToDoc1,
						otherDoc,
						nearRelatedDistancesToDocs[otherIdx],
						relatedWordsCache.second);
      }	else {
	throw std::invalid_argument("Unknown function: " + functionName);
      }
      float funcTime = (std::clock() - funcStart) / (float) CLOCKS_PER_SEC;
      if(verbose) {
	std::cout << doc1.nonZeros() << "\t"
		  << otherDoc.nonZeros() << "\t"
		  << funcTime << std::endl;
      }
    }

    numCorrect += (tripletDistances[0] < tripletDistances[1]? 1 : 0);
  }
  float experimentTime = (std::clock() - experimentStart) / (float) CLOCKS_PER_SEC;

  // Get error rate
  float accuracy = numCorrect / (float) tripletsDataset.size();

  // For some motive before ending function there is a peak in memory use
  relatedWordsCache.first.clear();
  return std::make_tuple(accuracy, preTime, experimentTime);
}

void
runTripletsExperiment(const std::string& tripletsFilePath,
		      const std::string& documentsFilePath,
		      const std::string& embeddingsFilePath,
		      int numClusters,
		      int maxIterations,
		      const std::string& functionName,
		      int r,
		      bool verbose)
{
  if(verbose) {
    std::cout << "Reading embeddings..." << std::endl;
  }
  Eigen::MatrixXf embeddings = Tools::getEmbeddings(embeddingsFilePath);

  if(verbose) {
    std::cout << "Reading triplets dataset..." << std::endl;
  }
  std::vector<std::tuple<int, int, int>> tripletsDataset
    = Tools::getTriplets(tripletsFilePath);

  if(verbose) {
    std::cout << "Reading documents..." << std::endl;
  }
  Eigen::SparseMatrix<int> documents
    = Tools::getTripletsDocuments(documentsFilePath, embeddings.cols());

  // Run experiment
  std::tuple<float, float, float> result =
    runRelatedDocumentPairsIdentification(tripletsDataset,
					  documents,
					  embeddings,
					  numClusters,
					  maxIterations,
					  functionName,
					  r,
					  verbose);
  float accuracy = std::get<0>(result);
  float preTime = std::get<1>(result);
  float experimentTime = std::get<2>(result);

  // Dump results to Console
  std::string SEP = "\t";
  std::string filesString
    = tripletsFilePath + SEP
    + documentsFilePath + SEP
    + embeddingsFilePath + SEP;
  std::string paramsString
    = functionName + SEP
    + std::to_string(r) + SEP;
  std::string timesString
    = std::to_string(preTime) + SEP
    + std::to_string(experimentTime) + SEP;
  std::cout << filesString
	    << paramsString
	    << timesString
	    << accuracy
	    << std::endl;
}

/* ---------- */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int
main(int argc, const char * argv[])
{
  std::string tripletsFilePath;
  std::string documentsFilePath;
  std::string embeddingsFilePath;
  std::string functionName;
  int r = -1;
  int numClusters = -1;
  int maxIterations = -1;
  bool verbose = false;

  char *mode = "triplets";

  // Parse cmd parameters
  for(int i = 1; i < argc; i+=2) {
    const char *cmd = argv[i];
    //printf("argv[%d]: %s\t%s\n", i, argv[i], argv[i+1]);
    if(strcmp(cmd, "--help") == 0) {
      printf("--trip <filepath>: %s\n\n", "Triplets dataset filepath");
      printf("--docs <filepath>: %s\n\n", "Documents filepath");
      printf("--emb <filepath>:  %s\n\n", "Embeddings filepath");
      printf("--num_clusters X:  %s\n\n", "Number of clusters");
      printf("--max_iter X:      %s\n\n",
	     "Maximum number of iterations during clustering");
      printf("--func <function>: %s\n\n", "Function to be used");
      printf("--r X:             %s\n\n", "Number of related words");
      exit(0);
    } else if(strcmp(cmd, "--verbose") == 0) {
      if (strcmp(argv[i+1], "true") == 0) verbose = true;
    } else if(strcmp(cmd, "--trip") == 0) {
      tripletsFilePath = argv[i+1];
    } else if(strcmp(cmd, "--docs") == 0) {
      documentsFilePath = argv[i+1];
    } else if(strcmp(cmd, "--emb") == 0) {
      embeddingsFilePath = argv[i+1];
    } else if(strcmp(cmd, "--func") == 0) {
      functionName = argv[i+1];
    } else if(strcmp(cmd, "--r") == 0) {
      r = std::stoi(argv[i+1]);
    } else if(strcmp(cmd, "--num_clusters") == 0) {
      numClusters = std::stoi(argv[i+1]);
    } else if(strcmp(cmd, "--max_iter") == 0) {
      maxIterations = std::stoi(argv[i+1]);
    } else {
      printf("Unknown option: %s\n", cmd);
      exit(1);
    }      
  }

  runTripletsExperiment(tripletsFilePath,
			documentsFilePath,
			embeddingsFilePath,
			numClusters,
			maxIterations,
			functionName,
			r,
			verbose);

  exit(0);
}
