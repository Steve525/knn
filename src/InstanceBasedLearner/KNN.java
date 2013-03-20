package InstanceBasedLearner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import toolkit.Matrix;

public class KNN {

	private Matrix _trainingExamples;
	private Matrix _trainingClassifications;
	private int _k;
	
	public KNN(int k, Matrix trainingExamples, Matrix trainingClassifications) {
		_k = k;
		_trainingExamples = trainingExamples;
		_trainingClassifications = trainingClassifications;
	}
	
	public KNN() { }
	
	public double classify(double[] values, boolean useClassification, boolean useWeighted) {
		Instance instance = new Instance(values);
		List<DistancePair> nearestNeighbors = getNearestNeighbors(getDistancesFromQuery(instance));
		if (useClassification && !useWeighted)
			return getUnweightedClassification(nearestNeighbors);
		if (useClassification && useWeighted)
			return getWeightedClassification(nearestNeighbors);
		if (!useClassification && !useWeighted)
			return getRegressionUnweighted(nearestNeighbors);
		if (!useClassification && useWeighted)
			return getRegressionWeighted(nearestNeighbors);
		throw new RuntimeException();
	}
	
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	// Regression -------------------------------------------------------------
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	private double getRegressionUnweighted(List<DistancePair> nearestNeighbors) {
		assert(_k == nearestNeighbors.size());
		double totalClassification = 0;
		for (DistancePair distancePair : nearestNeighbors) {
			totalClassification += _trainingClassifications.get((int)distancePair.getValue(), 0);
		}
		return (totalClassification / _k); 
	}
	
	private double getRegressionWeighted(List<DistancePair> nearestNeighbors) {
		double w = 0;
		double top = 0;
		for (DistancePair neighbor : nearestNeighbors) {
			top += (1 / (Math.pow(neighbor.getKey(), 2))) * _trainingClassifications.get(neighbor.getValue(), 0);
			w += 1 / (Math.pow(neighbor.getKey(), 2));
		}
		return (top / w);
	}
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	// Classification ---------------------------------------------------------
	// ------------------------------------------------------------------------
	private double getUnweightedClassification(List<DistancePair> nearestNeighbors) {
		assert(_k == nearestNeighbors.size());
		HashMap<Double, Integer> frequencies = new HashMap<Double, Integer>();
		for (DistancePair distancePair : nearestNeighbors) {
			int instance = (int) distancePair.getValue();
			Double vote = _trainingClassifications.get(instance, 0);
			Integer frequency = frequencies.get(vote);
			frequencies.put(vote, (frequency == null ? 1 : frequency + 1));
		}
		
		double mode = 0;
		int maxFreq = 0;
		
		for (Entry<Double, Integer> entry: frequencies.entrySet()) {
			int freq = entry.getValue();
			if (freq > maxFreq) {
				maxFreq = freq;
				mode = entry.getKey();
			}
		}
		
		return mode;
	}
	
	private double getWeightedClassification(List<DistancePair> nearestNeighbors) {
		HashMap<Double, HashSet<Double>>  classesToWeights = new HashMap<Double, HashSet<Double>>();
		double w = 0;
		for (DistancePair neighbor : nearestNeighbors) {
			w += 1 / (Math.pow(neighbor.getKey(), 2));
			double classification = _trainingClassifications.get(neighbor.getValue(), 0);
			HashSet<Double> weights = classesToWeights.get(classification);
			if (weights == null)
				weights = new HashSet<Double>();
			weights.add(neighbor.getKey());
			classesToWeights.put(classification, weights);
		}
		
		double top = 0;
		double bestClass = 0;
		
		for (Entry<Double, HashSet<Double>> entry : classesToWeights.entrySet()) {
			double currentTop = 0;
			Iterator<Double> iterator = entry.getValue().iterator();
			while (iterator.hasNext()) {
				currentTop += 1 / (Math.pow(iterator.next(), 2));
			}
			if (currentTop > top) {
				top = currentTop;
				bestClass = entry.getKey();
			}
		}
		
		return bestClass;
	}
	//-----------------------------------------------------------------------------------
	//-----------------------------------------------------------------------------------
	//-----------------------------------------------------------------------------------
	//-----------------------------------------------------------------------------------
	
	private class DistanceComparator implements Comparator<DistancePair> {

		@Override
		public int compare(DistancePair d0, DistancePair d1) {
			return ((java.lang.Double) d0.getKey()).compareTo((java.lang.Double) d1.getKey());
		}
		
	}
	
	private List<DistancePair> getNearestNeighbors(List<DistancePair> distances) {
		if (_k == 1)
			return distances.subList(0, 1);
		Collections.sort(distances, new DistanceComparator());
		return distances.subList(0, _k - 1);
	}
	
	private List<DistancePair> getDistancesFromQuery(Instance validationInstance) {
		List<DistancePair> distances = new ArrayList<DistancePair>();
		double distance;
		for (int i = 0; i < _trainingExamples.rows(); i++) {
			distance = 0;
			double[] trainingInstance = _trainingExamples.row(i);
			ArrayList<Double> testInstance = (ArrayList) validationInstance.getValues();
			for (int j = 0; j < trainingInstance.length; j++) {
				double trainingValue = trainingInstance[j];
				double validationValue = testInstance.get(j);
//				double trainingLabelType = _trainingExamples.valueCount(j);
//				System.out.println("VALUE COUNT: " + trainingLabelType);
//				if (trainingLabelType == 0)
				distance += Math.pow(trainingValue - validationValue, 2);
			}
//			System.out.println("DISTANCE: " + distance + ", " + i);
			distances.add(new DistancePair(distance, i));
		}
		return distances;
	}
	
	public class DistancePair {
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
		
		public void setKey(Double distance) {
			this.distance = distance;
		}
		
		public void setValue(Integer instance) {
			this.instance = instance;
		}
	}
}
