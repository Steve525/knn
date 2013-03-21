package InstanceBasedLearner;

import java.io.BufferedReader;
import java.io.FileReader;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class InstanceBasedLearner extends SupervisedLearner {

	private Matrix _trainingExamples;
	private Matrix _trainingLabels;
	private int _k;
	private KNN _knn;
	
	public InstanceBasedLearner(int k) {
		_k = k;
	}		
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		System.out.println("\t" + _k + "-nn learning algorithm.");
		_trainingExamples = features;
		_trainingLabels = labels;
		_knn = new KNN(_k, _trainingExamples, _trainingLabels);
//		_knn.reduceAllKnn();
		_knn.reduceVarSim();
//		System.out.println(_trainingExamples.rows());
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = _knn.classify(features, true, true);
	}


}
