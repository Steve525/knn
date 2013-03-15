package InstanceBasedLearner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import toolkit.Matrix;

public class KNN {

	private Matrix _trainingExamples;
	private int _k;
	
	public KNN(int k, Matrix trainingExamples) {
		_k = k;
		_trainingExamples = trainingExamples;
	}
	
	public double classify(double[] values) {
		Instance instance = new Instance(values);
		double classification = 0;
		List<Double> distances = getDistancesFromQuery(instance);
		List<Double> nearestNeightbors = getNearestNeighbors(distances);
		return classification;
	}
	
	private List<Double> getNearestNeighbors(List<Double> distances) {
		Collections.sort(distances);
		return distances.subList(distances.size() - _k, distances.size() - 1);
	}
	
	private List<Double> getDistancesFromQuery(Instance validationInstance) {
		List<Double> distances = new ArrayList<Double>();
		double distance;
		for (int i = 0; i < _trainingExamples.rows(); i++) {
			distance = 0;
			double[] trainingInstance = _trainingExamples.row(i);
			for (int j = 0; j < trainingInstance.length; j++) {
				// check if value is CONTINUOUS, NOMINAL, or MISSING
				double trainingValue = trainingInstance[j];
//				if (trainingValue)
			}
			
		}
		return distances;
	}
}
