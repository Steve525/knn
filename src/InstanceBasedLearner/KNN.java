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
import java.util.Set;

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
	
	public void reduceAllKnn() {
		List<Integer> bads = new ArrayList<Integer>();
		for (int i = 0; i < _trainingExamples.rows(); i++) {
			// Done with weighted non-regression.
			System.out.println("Instance #" + i);
			if (classify(_trainingExamples.row(i), true, true) != _trainingClassifications.get(i, 0)) {
				bads.add(i);
				System.out.println("Bad instance, will be removed: " + i);
			}
		}
		for (int i : bads) {
			_trainingExamples.remove(i);
			_trainingClassifications.remove(i);
		}
		_trainingExamples.print();
	}
	
	public void reduceVarSim() {
		for (int i = 0; i < _trainingExamples.rows(); i++) {
			Instance instance = new Instance(_trainingExamples.row(i));
			List<DistancePair> nearestNeighbors = 
				getNearestNeighbors(getDistancesFromQuery(instance));
			Set<Integer> integers = new HashSet<Integer>();
			for (DistancePair neighbor : nearestNeighbors) {
				integers.add(neighbor.getValue());
			}
			if (integers.size() == 1) {
				_trainingExamples.remove(i);
				_trainingClassifications.remove(i);
				System.out.println("Instance#: " + i);
			}
		}
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
	
	private List<DistancePair> getDistancesFromQuery(Instance queryinstance) {
		List<DistancePair> distances = new ArrayList<DistancePair>();
		double distance;
		for (int i = 0; i < _trainingExamples.rows(); i++) {
			distance = 0;
			double[] trainingInstance = _trainingExamples.row(i);
			ArrayList<Double> queryInstance = (ArrayList) queryinstance.getValues();
			for (int j = 0; j < trainingInstance.length; j++) {
				double trainingValue = trainingInstance[j];
				double queryValue = queryInstance.get(j);
				if (trainingValue == queryValue)
					distance += 0;
				// If the training value or the query value are missing, distance is '1'
				else if (trainingValue == Double.MAX_VALUE || queryValue == Double.MAX_VALUE)
					distance += 1;
				else {
					double trainingLabelType = _trainingExamples.valueCount(j);
					// If the training attribute is continuous... 
					if (trainingLabelType == 0)
						distance += (Math.abs(trainingValue - queryValue) / (4 * standardDeviation(j))); // change 0 to standard deviation!
					else {	// If the training attribute is nominal...
						distance += vdm(j, trainingValue, queryValue);
					}
//					distance += Math.pow(trainingValue - queryValue, 2);
				}
			}
			distances.add(new DistancePair(distance, i));
		}
		return distances;
	}
	
	// Returns standard deviation for values of attribute a
	private double standardDeviation(int a) {
		double u = 0;
		for (int i = 0; i < _trainingExamples.rows(); i++) {
			u += _trainingExamples.get(i, a);
		}
		u = u / _trainingExamples.rows();
		double sigma = 0;
		for (int i = 0; i < _trainingExamples.rows(); i++) {
			sigma += Math.pow(_trainingExamples.get(i, a) - u, 2);
		}
		sigma = sigma / _trainingExamples.rows();
		return sigma;
	}
	
	private double vdm(int a, double x, double y) {
		double vdm = 0;
		for (int i = 0; i < _trainingClassifications.valueCount(0); i++) {
			int countX = 0;
			int countY = 0;
			int countXC = 0;
			int countYC = 0;
			for (int j = 0; j < _trainingExamples.rows(); j++) {
				double val = _trainingExamples.get(j, a);
				double outputClass = _trainingClassifications.get(j, 0);
				if (val == x)
					countX++;
				if (val == y)
					countY++;
				if (val == x && (int)outputClass == i)
					countXC++;
				if (val == y && (int)outputClass == i)
					countYC++;
			}
			double ratioX = 0;
			if (countX != 0)
				ratioX = countXC / countX;
			double ratioY = 0;
			if (countY == 0)
				ratioY = countYC / countY;
			vdm += Math.pow(ratioX - ratioY, 2);
		}
		
		return vdm;
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
