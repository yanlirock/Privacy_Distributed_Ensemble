

import java.io.* ;
import java.lang.* ;
import java.util.* ;

public class SimpleVotingClassifier
{
	public int dimensionality ; // dimensionality of the space in which the data points live
	public double partition_size ; // size of the actual partition

	public int nb_of_participants; // number of participants in MABoost
	public int[] rule_attribute_index_participant ; // index of the attribute that will be used for Alice classifier
	public double[] threshold_participant ; // threshold of Alice decision stump
	public int[] lower_class_participant ; // class predicted by the abstention classifier between -infinity and the lower threshold of Alice

	public double weighted_error ; // weighted error occuring when using this abstention classifier on the partition set
	public double weighted_abstention_rate; // weighted abstention rate occuring when using this abstention classifier on the partition set
	public double weighted_good_rate; // weighted good rate occuring when using this abstention classifier on the partition set
	
	public double penalty_misclassification; // penalty applied to a datapoint when misclassification happens
	public double penalty_abstention; // penalty applied to a datapoint for an abstention
	public double penalty_good_classification; // penalty applied to a datapoint for a good classification
	public double weight_classifier ; // weight of the actual classifier

	// constructor
	// the abstention classifier will be constructed from the merging of the weak classifiers of Alice and Bob
	// the index of the attribute of Alice and Bob rule is chosen at random among the possible ones
	public SimpleVotingClassifier(int d, Vector v)
	{
		int random_number;
		int z,y;
		Random r = new Random(); // pseudo-random generator
		int temp_random_attribute;
		MAWeakClassifier[] participant_weak_classifier;
		int[] attribute_considered;
		int attribute_min;
		double weighted_error_min;

		// initialisation of the variables
		dimensionality = d;
		partition_size = -1;
		nb_of_participants = v.size();
		rule_attribute_index_participant = new int[nb_of_participants];
		threshold_participant = new double[nb_of_participants];
		lower_class_participant = new int[nb_of_participants];
		attribute_considered = new int[nb_of_participants];
		participant_weak_classifier = new MAWeakClassifier[nb_of_participants];
		for (z=0; z<nb_of_participants; z++)
			participant_weak_classifier[z] = (MAWeakClassifier) v.elementAt(z);		

		// choice of Alice attribute for the decision stump at random among the valid ones
		for (z=0; z<nb_of_participants; z++)
		{
			temp_random_attribute = r.nextInt(dimensionality);
			while(participant_weak_classifier[z].array_weighted_error[temp_random_attribute] >= 0.5)
				temp_random_attribute = r.nextInt(dimensionality);
			//attribute_considered[z] = temp_random_attribute;	
			rule_attribute_index_participant[z] = temp_random_attribute;
			threshold_participant[z] = participant_weak_classifier[z].array_rule_threshold[temp_random_attribute];
			lower_class_participant[z] = participant_weak_classifier[z].array_rule_class[temp_random_attribute];
		}

		/*for (z=0; z<nb_of_participants; z++)
		{
			weighted_error_min = 1000;
			attribute_min = -1;
			for (y=0; y<nb_of_participants; y++)
			{
				if(participant_weak_classifier[z].array_weighted_error[attribute_considered[y]] < weighted_error_min)
				{
					weighted_error_min = participant_weak_classifier[z].array_weighted_error[attribute_considered[y]];
					attribute_min = attribute_considered[y];
				}
				rule_attribute_index_participant[z] = attribute_min;
				threshold_participant[z] = participant_weak_classifier[z].array_rule_threshold[attribute_min];
				lower_class_participant[z] = participant_weak_classifier[z].array_rule_class[attribute_min];
			}	
		}*/

		weighted_error = -1000;
		weighted_abstention_rate = -1000;
		weighted_good_rate = -1000;
		penalty_misclassification = -1000;
		penalty_abstention = -1000;
		penalty_good_classification = -1000;
		weight_classifier = -1000;
	}

	// classify the point given as parameter by using the merged classifier
	// if the classifier of Alice and Bob from which the merged classifier is composed agreed then their predictions is returned as output
	// otherwise the merged classifier abstains by returning 0
	public int classify_point(DataPoint p)
	{
		int value_return;
		int sum_classification;
		int result_classifier_participant;
		int z;
		
		sum_classification = 0;
		
		for (z=0; z<nb_of_participants; z++)
		{
			if (p.coordinates[rule_attribute_index_participant[z]] < threshold_participant[z])
				result_classifier_participant = lower_class_participant[z];
			else 
			{
				if (lower_class_participant[z] == -1)
					result_classifier_participant = 1;
				else
					result_classifier_participant = -1;
			} 
			sum_classification += result_classifier_participant;
		}

		if (sum_classification == 0)
			value_return = 0;
		else
		{	
			if (sum_classification > 0)
				value_return = 1;
			else
				value_return = -1;
		}

		return value_return;
	}

	// compute the weighted error, the weighted abstention rate and the weighted good rate of the merged classifier
	public void compute_weighted_rates(DataPoint[] partition_set)
	{
		int x;
		int temp_class;

		weighted_error = 0;
		weighted_abstention_rate = 0;
		weighted_good_rate = 0;
		partition_size = partition_set.length;
		for(x=0; x<partition_size; x++)
		{
			temp_class = classify_point(partition_set[x]);
			if (temp_class == 0)
				weighted_abstention_rate += partition_set[x].weight;
			else
			{
				if (temp_class == partition_set[x].point_class)
					weighted_good_rate += partition_set[x].weight;
				else
					weighted_error += partition_set[x].weight;
			}	 
			
		} 
		
		
	}

	// compute the misclassification penalty, the good classification "penalty" and the abstention penalty of the classifier
	public void compute_penalties()
	{
		double small_appropriate_constant;

		small_appropriate_constant = 1/partition_size;
		penalty_misclassification = 1/((2* weighted_error)+ (weighted_abstention_rate * (double) Math.sqrt(((weighted_error+small_appropriate_constant)/(weighted_good_rate+small_appropriate_constant)))));
		penalty_good_classification = 1/((2* weighted_good_rate)+ (weighted_abstention_rate * (double) Math.sqrt(((weighted_good_rate+small_appropriate_constant)/(weighted_error+small_appropriate_constant)))));
		penalty_abstention = 1/(weighted_abstention_rate+ (2* (double) Math.sqrt(weighted_good_rate*weighted_error)));
	}

	// compute the weight of the classifier by using the weighted error and the weighted good rate
	public void compute_weight_classifier()
	{
		double result;
		double small_appropriate_constant;

		small_appropriate_constant = 1/partition_size;
		result = 0.5 * Math.log((weighted_good_rate+ small_appropriate_constant)/(weighted_error+small_appropriate_constant));
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

		
		
