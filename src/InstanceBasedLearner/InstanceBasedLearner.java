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
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		_k = 3;
		_trainingExamples = features;
		_trainingClassifications = labels;
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		KNN knn = new KNN(_k, _trainingExamples, _trainingClassifications);
		labels[0] = knn.classify(features, true, false);
	}

}
