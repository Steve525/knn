package InstanceBasedLearner;

import java.io.BufferedReader;
import java.io.FileReader;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class InstanceBasedLearner extends SupervisedLearner {

	private Matrix _trainingExamples;
	private Matrix _trainingClassifications;
	private int _k;
	static double MISSING = Double.MAX_VALUE;

	public InstanceBasedLearner(int k) {
		_k = k;
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		_trainingExamples = features;
		_trainingClassifications = labels;
		System.out.println("\t" + _k + "-nn learning algorithm.");
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		KNN knn = new KNN(_k, _trainingExamples, _trainingClassifications);
		labels[0] = knn.classify(features, false, false);
	}

}
