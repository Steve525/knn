package InstanceBasedLearner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 	An instance will contain a list of attribute values
 * @author Stephen Clarkson
 *
 */
public class Instance {
	
	List<Double> _values;
	
	public Instance(double[] values) {
		_values = new ArrayList<Double>();
		for (double val : values) {
			_values.add(val);
		}
	}
	
	public List<Double> getValues() {
		return _values;
	}

}
