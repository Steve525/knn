package InstanceBasedLearner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
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
		List<DistancePair> nearestNeightbors = getNearestNeighbors(getDistancesFromQuery(instance));
		
		return classification;
	}
	
	private class DistanceComparator implements Comparator<DistancePair> {

		@Override
		public int compare(DistancePair d0, DistancePair d1) {
			return ((java.lang.Double) d0.getKey()).compareTo((java.lang.Double) d1.getKey());
		}
		
	}
	
	private List<DistancePair> getNearestNeighbors(List<DistancePair> distances) {
		Collections.sort(distances, new DistanceComparator());
		return distances.subList(distances.size() - _k, distances.size() - 1);
	}
	
	private List<DistancePair> getDistancesFromQuery(Instance validationInstance) {
		List<DistancePair> distances = new ArrayList<DistancePair>();
		double distance;
		for (int i = 0; i < _trainingExamples.rows(); i++) {
			distance = 0;
			double[] trainingInstance = _trainingExamples.row(i);
			for (int j = 0; j < trainingInstance.length; j++) {
				double trainingValue = trainingInstance[j];
				double validationValue = validationInstance.getValues().get(j);
				// check if value is CONTINUOUS, NOMINAL, or MISSING
				distance += Math.pow(trainingValue - validationValue, 2);
			}
			distances.add(new DistancePair(distance, i));
		}
		return distances;
	}
	
	private class DistancePair {
		private Double distance;
		private Integer instance;
		
		public DistancePair(Double distance, Integer instance) {
			this.distance = distance;
			this.instance = instance;
		}
		
		public Double getKey() {
			return this.distance;
		}
		
		public Integer getValue() {
			return this.instance;
		}
	}
}
