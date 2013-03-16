package InstanceBasedLearner;

import java.util.*;

import org.junit.Before;
import org.junit.Test;

import toolkit.Matrix;

import InstanceBasedLearner.KNN.DistancePair;


public class KNNMethodTests {
	
	private Matrix _trainingExamples;
	private Matrix _trainingClassifications;
	private int _k;
	
	@Before
	public void setUp() {
		String trainingExamples = "C:\\Users\\Steve\\Documents\\478\\knn\\training-examples\\mt_train.arff";
		
	}
	
	@Test
	public void testGetMostCommonClass() {
		List<DistancePair> nearestNeighbors = new ArrayList<DistancePair>();
		KNN knn = new KNN();
		DistancePair d1 = knn.new DistancePair(1.1, 2);
		DistancePair d2 = knn.new DistancePair(31.0, 1);
		DistancePair d3 = knn.new DistancePair(5.3, 3);
		DistancePair d4 = knn.new DistancePair(6.4, 4);
		nearestNeighbors.add(d1);
	}
}
