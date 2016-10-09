// MAWeakClassifier.java
// holds the information regarding the MABoost weak classifier, here we use the decision stumps
// Sebastien Gambs

import java.io.* ;
import java.lang.* ;

public class MAWeakClassifier
{
	public int dimensionality ; // dimensionality of the space in which the data points live
	
	public double[] array_rule_threshold ; // array containing the thresholds for the rules of the decision stumps
					      // remark: there is an optimal decision stump for each dimension
	public int[] array_rule_class ; // array containing the class determined by the threshold of decision stumps
				        // if attribute of the rule index is below this threshold the object is classify as belonging to the rule class
  	
	public double[] array_weighted_error ; // array containing the weighted error occuring when using a decision stump to classify
	public double weight_classifier ; // weight of the actual classifier

	// constructor
	public MAWeakClassifier(int d)
	{
		int x;

		// initialisation of the variables
		dimensionality = d;
		
		array_rule_threshold = new double[dimensionality];
		array_rule_class = new int[dimensionality];
		array_weighted_error = new double[dimensionality];
		for (x=0; x<dimensionality; x++)
		{
			array_rule_threshold[x] = -100;
			array_rule_class[x] = -100;	
			array_weighted_error[x] = 100;
		} 

		weight_classifier = -1;
	}

	// find the best decision stumps for the current distribution of the data and for each attribute
	public void find_best_decision_stumps(DataPoint[] partition_set)
	{
		double[] temp_results;
		DataPoint[] temp_p;
		int x;
		
		// for each attribute, sort the objects according to this dimension and find the best decision stump for this particular dimension
		for (x=0; x<dimensionality; x++)
		{
			temp_p = insertion_sort(partition_set,x);
			if (is_array_sorted(temp_p,x) == false)
				System.out.println("Problem: the array should be sorted");
			temp_results = find_best_decision_stump(temp_p,x);
			array_rule_threshold[x] = temp_results[0];
			array_rule_class[x] = (int)temp_results[1];
			array_weighted_error[x] = temp_results[2];
			
		} 
	}

	// find the best decision stump for the current distribution of the weight and a specific dimension
	// return the threshold found for this rule, the class located below the threshold and the resulting weighted error
	public double[] find_best_decision_stump(DataPoint[] p, int index_a)
	{
		double[] value_return;
		int p_size;
		double best_decision_stump_threshold;
		double best_weighted_error;
		int best_decision_stump_class;
		double current_decision_stump_threshold;
		double positive_weighted_error;
		double negative_weighted_error;
		double current_weighted_error;
		int current_decision_stump_class;
		int x,y;
		
		int nb_thresholds, current_threshold;
		double[] temp_threshold;	
		
		p_size = p.length;
		best_weighted_error = 100;
		best_decision_stump_threshold = -100;
		best_decision_stump_class = 100;

		nb_thresholds = 0;
		for (x=0; x< (p_size-1); x++)
		{
			if (p[x].coordinates[index_a] != p[x+1].coordinates[index_a])
				nb_thresholds++;
		}

		temp_threshold = new double[nb_thresholds];
		current_threshold = 0;
		for (x=0; x< (p_size-1); x++)
		{
			if (p[x].coordinates[index_a] != p[x+1].coordinates[index_a])
			{
				temp_threshold[current_threshold] = (p[x].coordinates[index_a] + p[x+1].coordinates[index_a])/2;
				current_threshold++;
			}
		}

		// for each threshold, compute the weighted error resulting in placing the threshold at this point
		// in both cases where the lower class is positive and negative
		for(x=0; x< nb_thresholds; x++)
		{
			current_decision_stump_threshold = temp_threshold[x];

			positive_weighted_error = 0;
			negative_weighted_error = 0;
			// compute the weighted error resulting in placing the threshold as the current position
			// and predicting +1 below this threshold and -1 above
			for (y=0; y<p_size; y++)
			{
				if ((p[y].coordinates[index_a] < current_decision_stump_threshold) && (p[y].point_class == -1))
					positive_weighted_error += p[y].weight;
				if ((p[y].coordinates[index_a]>= current_decision_stump_threshold) && (p[y].point_class == 1))
					positive_weighted_error += p[y].weight;
			}
			// compute the weighted error resulting in placing the threshold as the current position
			// and predicting -1 below this threshold and +1 above
			for (y=0; y<p_size; y++)
			{
				if ((p[y].coordinates[index_a] < current_decision_stump_threshold) && (p[y].point_class == 1))
					negative_weighted_error += p[y].weight;
				if ((p[y].coordinates[index_a]>= current_decision_stump_threshold) && (p[y].point_class == -1))
					negative_weighted_error += p[y].weight;
			}
			
			// choose the best option between the positive and negative error
			if (positive_weighted_error < negative_weighted_error)
			{
				current_weighted_error = positive_weighted_error;
				current_decision_stump_class = 1;
			}
			else
			{
				current_weighted_error = negative_weighted_error;
				current_decision_stump_class = -1;
			}
			
			if (current_weighted_error < best_weighted_error)
			{
				best_weighted_error = current_weighted_error;
				best_decision_stump_threshold = current_decision_stump_threshold;
				best_decision_stump_class = current_decision_stump_class;
			}

		} 
			
		value_return = new double[3];
		value_return[0] = best_decision_stump_threshold;
		value_return[1] = best_decision_stump_class;
		value_return[2] = best_weighted_error;
		
		return value_return;
	}

	// sort a partition by insertion based on the attribute whose index is given as a parameter
	// and return the sorted array
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

		
		
