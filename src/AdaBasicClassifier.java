// AdaBasicClassifier.java
// holds the information regarding the basic classifier used for AdaBoost, here we use the decision stumps
// Sebastien Gambs

import java.io.* ;
import java.lang.* ;

public class AdaBasicClassifier
{
	public int dimensionality ; // dimensionality of the space in which the data points live
	public double partition_size ; // size of the actual partition

	public int rule_attribute_index ; // index of the attribute that will be used for the decision stump
	public double rule_threshold ; // threshold in the rule for the decision stump
	public int lower_class ; // class predicted by the basic classifier below the rule threshold
	public int upper_class ; // class predicted by the basic classifier above the rule threshold
	
	public double weighted_error ; // weighted error occuring when using this decision stump to classify the partition set
	public double weighted_good_rate; // weighted error occuring when using this decision stump to classify the partition set
	
	public double penalty_misclassification; // penalty applied to a datapoint when misclassification happens
	public double penalty_good_classification; // penalty applied to a datapoint for a good classification
	public double weight_classifier ; // weight of the actual classifier

	// constructor
	public AdaBasicClassifier(int d, AdaWeakClassifier a)
	{
		int temp_best_attribute;
		double min_weighted_error;
		int x;

		// initialisation of the variables
		dimensionality = d;
		partition_size = -1;
		
		temp_best_attribute = 0;
		min_weighted_error = a.array_weighted_error[0];
		for (x=1; x<dimensionality; x++)
		{
			if (a.array_weighted_error[x] < min_weighted_error)
			{
				min_weighted_error = a.array_weighted_error[x];
				temp_best_attribute = x;	
			}
		}
		rule_attribute_index = temp_best_attribute;
		rule_threshold = a.array_rule_threshold[rule_attribute_index];
		lower_class = a.array_rule_class[rule_attribute_index];
		if (lower_class == -1)
			upper_class = 1;
		else
			upper_class = -1;

		weighted_error = -1000;
		weighted_good_rate = -1000;
		penalty_misclassification = -1000;
		penalty_good_classification = -1000;
		weight_classifier = -1000;
	}

	// classify the point given as parameter by using the rule of the basic classifier
	// value of attribute < rule threshold => lower class
	// rule threshold <= value of attribute => upper class
	public int classify_point(DataPoint p)
	{
		int value_return;

		if (p.coordinates[rule_attribute_index] < rule_threshold)
			value_return = lower_class;
		else
			value_return = upper_class;
	
		return value_return;
	}

	// compute the weighted error and the weighted good rate of the basic classifier
	public void compute_weighted_rates(DataPoint[] partition_set)
	{
		int x;
		int temp_class;

		partition_size = partition_set.length;

		weighted_error = 0;
		weighted_good_rate = 0;
		for(x=0; x<partition_size; x++)
		{
			temp_class = classify_point(partition_set[x]);
			if (temp_class == partition_set[x].point_class)
				weighted_good_rate += partition_set[x].weight;
			else
				weighted_error += partition_set[x].weight;	 
		} 	
		
	}

	// compute the misclassification penalty and the good classification penalty of the basic classifier
	public void compute_penalties()
	{
		int x;
		double small_appropriate_constant;

		small_appropriate_constant = 1/partition_size;


		penalty_misclassification = (double) (1/ ((2*weighted_error)+small_appropriate_constant));
		penalty_good_classification = (double) (1/((2 * (1- weighted_error)) +small_appropriate_constant));	
	}

	// compute the weight of the classifier by using the weighted error and the weighted good rate
	public void compute_weight_classifier()
	{
		double result;
		double small_appropriate_constant;

		small_appropriate_constant = 1/partition_size;
		
		result = 0.5 * Math.log(((1-weighted_error)+ small_appropriate_constant)/(weighted_error+small_appropriate_constant));
		weight_classifier = (double) result;
		//System.out.println("Weighted error : " + weighted_error + " weight of the classifier : " + weight_classifier);
	}

	// sort a partition by insertion based on the attribute whose index is given as a parameter
	public DataPoint[] insertion_sort(DataPoint[] p, int index_a)
	{
		DataPoint[] value_return;
		int p_size;
		int i,j;
		DataPoint temp_point;

		p_size = p.length;
		value_return = new DataPoint[p_size];
		for (i=0 ; i<p_size; i++)
			value_return[i] = p[i].cloneDataPoint();

		for (i=1 ; i<p_size; i++)
		{
			if (value_return[i].coordinates[index_a] < value_return[i-1].coordinates[index_a])
			{
				j = i;
				while ((j >= 1) && (value_return[j].coordinates[index_a] < value_return[j-1].coordinates[index_a]))
				{
					temp_point = value_return[j-1].cloneDataPoint();
					value_return[j-1] = value_return[j];
					value_return[j] = temp_point;
					j--;
				}
			
			}
		}	
		
		return value_return;	
	}

	// check if a partition is sorted according to the attribute whose index is given as parameter
	public boolean is_array_sorted(DataPoint[] p,int index_a)
	{
		boolean value_return;
		int p_size;
		int x;
		
		p_size = p.length;
		value_return = true;
		x = 1;
		while((value_return == true) && (x < p_size))
		{
			if (p[x-1].coordinates[index_a] > p[x].coordinates[index_a])
				value_return = false;
			x++; 							
		}
	
		return value_return;
	}
	
	// check if the weights of the data points are normalized
	public boolean check_weight_normalization(DataPoint[] p)
	{
		boolean value_return;
		int p_size;
		double sum;
		int x;
		
		p_size = p.length;
		sum = 0;
		for (x=0; x<p_size; x++)
			sum += p[x].weight;
		System.out.println("Somme des poids : " + sum);		
		
		value_return = true;
		return value_return;
	}
	
}

		
		
