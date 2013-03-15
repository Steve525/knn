package InstanceBasedLearner;

import java.io.BufferedReader;
import java.io.FileReader;

import toolkit.Matrix;
import toolkit.SupervisedLearner;

public class InstanceBasedLearner extends SupervisedLearner {

	private Matrix _trainingExamples;
	private int _k;
	static double MISSING = Double.MAX_VALUE;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		_k = 3;
		final String instanceFile = "training-examples/????";
		_trainingExamples = new Matrix();
		_trainingExamples.loadArff(instanceFile);
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		KNN knn = new KNN(_k, _trainingExamples);

	}

}
