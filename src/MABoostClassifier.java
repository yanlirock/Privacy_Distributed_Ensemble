// MABoostClassifier.java
// manage the MABoost Classifier with decision stumps for 3 or more participants
// Sebastien Gambs

import java.io.* ;
import java.util.* ;

public class MABoostClassifier
{
	public int dimensionality ; // dimensionality of the space
	public int nb_of_iterations ; // number of iterations of the boosting process
	public int nb_of_participants ; // number of participants involved in MABoost (usually greater or equal to 3)
	public int nbdatapoints ; // number of points in the training set
	public int[] nbparticipantpoints; //  number of points in the participant training set
	public int nbtestpoints ; // number of points in the test set
	//public SimpleVotingClassifier[] common_weak_classifier_array ; // common weak classifiers stored in an array
	public AdaBasicClassifier[][] common_weak_classifier_array ; 
	public AdaWeakClassifier[][] participant_weak_classifier_array; // each participant weak classifiers stored in an array
	public double[] training_error; // array storing the overall training error of the MABoost classifier at each iteration
	public double[][] participant_training_error; // array storing the training error on the participants set of the MABoost classifier at each iteration
	public double[] test_error; // array storing the test error of the MABoost classifier at each iteration

	public String data_filename ; // name of the file containing the data
	public DataPoint[] data_set ; // array containing the points of the data set
	public DataPoint[] test_set ; // array containing the points of the test set
	public DataPoint[][] participant_set; // array containing the participants data set
	public int nb_good_training ; // number of points well classified on the training set
	public int nb_bad_training ; // number of points misclassified on the training set
	public int nb_good_test ; // number of points well classified on the training set
	public int nb_bad_test ; // number of points misclassified on the training set

	public int number_of_validations; // number of validations used
	public int size_in_blocks_validation; // size in blocks of a validation
	public double[][] val_training_error; // array storing the overall training error of the MABoost classifier at each iteration and each validation
	public double[][][] val_participant_training_error; // array storing the training error on Alice's set of the MABoost classifier at each iteration and each validation
	public double[][] val_test_error; // array storing the test error of the MABoost classifier at each iteration and each validation
	public DataPoint[][] val_test_set ; // array containing the points of the test set for each validation
	public DataPoint[][][] val_participant_set; // array containing the dataset of Alice for each validation

	public double real_fold_training_error[];
	public double real_fold_test_error[];
	public int real_fold_iteration_optimal[];
	public double[][] real_fold_validation_error;

	// constructor
	public MABoostClassifier(String f_data, int nb_iterations, int nb_participants)
	{
		data_filename = new String(f_data);
		nb_of_iterations = nb_iterations;
		nb_of_participants = nb_participants;
		number_of_validations = 2;
	}

	// run the MABoost multiple participants classification
	public void runClassification()
	{
		String training_error_representation;
		String participant_training_error_representation;
		String test_error_representation;
        String global_results_representation;
		DataPoint[] temp_data_array;
		int x,y;

		// compute the size of the dataset, load the dataset and construct Alice's, Bob's and the test set from this dataset
		compute_data_size();
		load_dataset();
		
		construct_validation_sets();
		for (x=0; x < number_of_validations; x++)
		{
			System.out.println("Validation number : "+ (x+1));
			instance_particular_validation_set(x);
			constructMABoostClassifier();
			training_error_representation = print_training_error();
			for (y=0; y <nb_of_participants; y++)
			{	
				participant_training_error_representation = print_participant_training_error(y);
				save_participant_error_validation(participant_training_error_representation,y,x);
			}
			test_error_representation = print_test_error();
			save_training_error_validation(training_error_representation,x);
			save_test_error_validation(test_error_representation,x);
			save_results_validation(x);
		}
		compute_global_validation_results();
		construct_real_fold_results();
		compute_global_real_fold_results();
		global_results_representation = print_global_results();
		save_global_results(global_results_representation);

		// save the overall training error, Alice's training error, Bob's training error and the test error on files
		// respectively in "training_error.txt", "alice_error.txt", "bob_error.txt" and "test_error.txt"
		training_error_representation = print_training_error();
		test_error_representation = print_test_error();
		//bound_representation = print_bound();
		//prod_bound_representation = print_prod_bound();
		save_training_error(training_error_representation);
		save_test_error(test_error_representation);

		/*classify_training_set();
		System.out.println("Number of points well classified in the training set : " + nb_good_training);
		System.out.println("Number of points misclassified in the training set : " + nb_bad_training);
		classify_test_set();
		System.out.println("Number of points well classified in the test set : " + nb_good_test);
		System.out.println("Number of points misclassified in the test set : " + nb_bad_test);
		System.out.println(nb_bad_training + " " + nb_bad_test);*/
		//print_training_set();
		//print_test_set();
	}

	// contruct the MABoost Classifier
	public void constructMABoostClassifier()
	{
		double initial_weight;
		int x,y,z;
		int i;
		int temp_att, temp_class;
		DataPoint[] temp_partition_set;
		int total_point_participant;
		double temp_weighted_abstention, temp_weighted_error, temp_weighted_good;
		Vector temp_weak_classifier_participants;

		// initialisation of the variables of the MABoost classifier
		common_weak_classifier_array = new AdaBasicClassifier[nb_of_participants][];
		for (z=0; z<nb_of_participants; z++)
			common_weak_classifier_array[z] = new AdaBasicClassifier[nb_of_iterations];
		participant_weak_classifier_array = new AdaWeakClassifier[nb_of_participants][];
		for (z=0; z<nb_of_participants; z++)
			participant_weak_classifier_array[z] = new AdaWeakClassifier[nb_of_iterations];
		training_error = new double[nb_of_iterations];
		participant_training_error = new double[nb_of_participants][];
		for (z=0; z<nb_of_participants; z++)
			participant_training_error[z] = new double[nb_of_iterations];
		test_error = new double[nb_of_iterations];
	
		// the initial weight of all the datapoints is set to 1 divided by the number of datapoints to indicate
		// that a priori all the datapoints have the same difficulty 
		//total_point_participant = 0;
		//for (z=0; z<nb_of_participants; z++)
			//total_point_participant += nbparticipantpoints[z];
		//initial_weight = 1/((double) total_point_participant);
		for (z=0; z<nb_of_participants; z++)
		{
			initial_weight = 1/((double) nbparticipantpoints[z]);
			for (x=0; x < nbparticipantpoints[z]; x++)
			{
      				participant_set[z][x].weight = initial_weight;
			}
		}
		//temp_partition_set= new DataPoint[total_point_participant];

		i = 0;
		while (i < nb_of_iterations)
		{
      			System.out.println("Iteration number : " + i);
			// check_weight_normalization();
			// for both Alice and Bob find the best decision stumps regarding each dimension
			for (z=0; z<nb_of_participants; z++)
			{
				participant_weak_classifier_array[z][i] = new AdaWeakClassifier(dimensionality);
				participant_weak_classifier_array[z][i].find_best_decision_stumps(participant_set[z]);
				common_weak_classifier_array[z][i] = new AdaBasicClassifier(dimensionality,participant_weak_classifier_array[z][i]);
			}
			temp_att = 0;
			if (temp_att == -1)
				i = nb_of_iterations;
			else
			{
				// put Alice and Bob datasets together in order to compute the error of the classifer on the overall training set
				/*x = 0;
				for (z=0; z<nb_of_participants; z++)
				{
					for (y = 0; y <nbparticipantpoints[z]; y++)
					{
						temp_partition_set[x] = participant_set[z][y].cloneDataPoint();
						x++;
					}
				}*/
				
				//temp_weak_classifier_participants = new Vector();
				//for (z=0; z<nb_of_participants; z++)
					//temp_weak_classifier_participants.addElement(participant_weak_classifier_array[z][i]);
				// construct the merged abstention classifier from Alice and Bob weak classifiers
				//common_weak_classifier_array[i] = new SimpleVotingClassifier(dimensionality,temp_weak_classifier_participants);
				// compute the weighted rates, the penalties and the weight of the merged classifier 
				for (z=0; z<nb_of_participants; z++)
				{
					common_weak_classifier_array[z][i].compute_weighted_rates(participant_set[z]);
					common_weak_classifier_array[z][i].compute_penalties();
					common_weak_classifier_array[z][i].compute_weight_classifier();
				}
				// update the weights of the participant points
				for (z=0; z<nb_of_participants; z++)
				{
					for (x=0; x < nbparticipantpoints[z]; x++)
					{
      						temp_class = common_weak_classifier_array[z][i].classify_point(participant_set[z][x]);
						//if (temp_class == 0)
						//	participant_set[z][x].weight = participant_set[z][x].weight * common_weak_classifier_array[z][i].penalty_abstention;
						//else
						//{	
							if (participant_set[z][x].point_class == temp_class)
								participant_set[z][x].weight = participant_set[z][x].weight * common_weak_classifier_array[z][i].penalty_good_classification;
							else
								participant_set[z][x].weight = participant_set[z][x].weight * common_weak_classifier_array[z][i].penalty_misclassification;
						//}
					}
				}
				// compute and store all the errors (training and test)
				//temp_weighted_abstention = common_weak_classifier_array[i].weighted_abstention_rate;
				//temp_weighted_error = common_weak_classifier_array[i].weighted_error;
				//temp_weighted_good = common_weak_classifier_array[i].weighted_good_rate;
				i++;
				for (z=0; z<nb_of_participants; z++)
				{
					participant_training_error[z][i-1] = compute_training_error(participant_set[z],z,i);
				}
				test_error[i-1] = compute_test_error(test_set,i);
				training_error[i-1] = 0;
				for (z=0; z<nb_of_participants; z++)
					training_error[i-1] += participant_training_error[z][i-1];
				training_error[i-1] = training_error[i-1]/nb_of_participants;
			}
		}	
	}

	// construct the participants' datasets and the test set for each validation
	public void construct_validation_sets()
	{
		DataPoint[] temp_data_array;
		int[] size_block_data;
		double temp_block_size;
		double last_block_size;
		int temp_test_size;
		int[] temp_participant_size;
		int x,y,z;
		int current_position_dataset, current_position_training;
		int a,b,t;
		int[] participant_count_block;
	        boolean[][] is_participant_validation_block;
		int current_participant;
		DataPoint[] temp_training_set;
		int temp_training_size;

		// initialize the array that will contain the different errors for each validation
		val_training_error = new double[number_of_validations][nb_of_iterations];
		//for (x=0; x<nb_of_participants; x++)
			//val_participant_training_error = new double[x][number_of_validations][nb_of_iterations];
		val_participant_training_error = new double[nb_of_participants][number_of_validations][nb_of_iterations];
		val_test_error = new double[number_of_validations][nb_of_iterations];

		// initialize a temporary array
		temp_data_array = new DataPoint[nbdatapoints]; 
		for (x=0; x < nbdatapoints; x++)
      			temp_data_array[x] = data_set[x].cloneDataPoint();
		
		size_block_data = new int[number_of_validations];
		
		// initialize the array which will contains the participants' dataset and the test dataset during the different validation
		//for (x=0; x<nb_of_participants; x++)
			//val_participant_set = new DataPoint[x][number_of_validations][];
		val_participant_set = new DataPoint[nb_of_participants][number_of_validations][];
		val_test_set = new DataPoint[number_of_validations][];

		// compute the size of the different blocks
		temp_block_size = Math.round(((double)nbdatapoints)/number_of_validations);
		last_block_size = (double) nbdatapoints - (temp_block_size * (number_of_validations-1));
		for (x = 0; x<number_of_validations-1; x++)
			size_block_data[x] = (int) temp_block_size;
		size_block_data[number_of_validations-1] = (int) last_block_size;
		
		// construct the participants' dataset and the test dataset for each validation step
		for (x = 0; x<number_of_validations; x++)
		{
			current_position_dataset = 0;
			current_position_training = 0;
			temp_training_size = nbdatapoints - size_block_data[x];
			temp_training_set = new DataPoint[temp_training_size];
            val_test_set[x] = new DataPoint[size_block_data[x]];

			for(y=0; y<number_of_validations; y++)
			{
				if (y == x)
				{
					for (z=0; z<size_block_data[y]; z++)
					{
						val_test_set[x][z] = temp_data_array[current_position_dataset].cloneDataPoint();
						current_position_dataset++;
					}
				}
				else
				{
					for (z=0; z<size_block_data[y]; z++)
					{
						temp_training_set[current_position_training] = temp_data_array[current_position_dataset].cloneDataPoint();
						current_position_dataset++;
						current_position_training++;
					}
				}
			}
			construct_participants_dataset(temp_training_set, nb_of_participants,x);
		}
		
	}

	// construct the training set and the validation set for each validation
	public void construct_participants_dataset(DataPoint[] d_set, int nb_part, int num_fold)
	{
		DataPoint[] temp_data_array;
		int[] size_block_data;
		double temp_block_size;
		double last_block_size;
		int x,y;
		int current_position;
	    int size_d_set;

		// initialize a temporary array
		size_d_set = d_set.length;
		temp_data_array = new DataPoint[size_d_set]; 
		for (x=0; x < size_d_set; x++)
			temp_data_array[x] = d_set[x].cloneDataPoint();
		
		size_block_data = new int[nb_part];
		
		// compute the size of the different blocks
		temp_block_size = Math.round(((double)size_d_set)/nb_part);
		last_block_size = (double) size_d_set - (temp_block_size * (nb_part-1));
		for (x = 0; x<nb_part-1; x++)
			size_block_data[x] = (int) temp_block_size;
		size_block_data[nb_part-1] = (int) last_block_size;
		
		// construct the training and the validation dataset for each validation step
		current_position =0;
		for (x = 0; x<nb_part; x++)
		{
			val_participant_set[x][num_fold] = new DataPoint[size_block_data[x]];

			for (y=0; y<size_block_data[x]; y++)
			{
				val_participant_set[x][num_fold][y] = d_set[current_position].cloneDataPoint();
				current_position++;
			}
		}
		
	}

	// construct the arrays containing the final results of each fold
	public void construct_real_fold_results()
	{
		int x,y;

		// declaration of the arrays
		real_fold_training_error = new double[number_of_validations];
		real_fold_test_error = new double[number_of_validations];
		real_fold_iteration_optimal = new int[number_of_validations];
		real_fold_validation_error = new double[number_of_validations][nb_of_iterations];

		// initialization of the arrays
		for (x=0; x<number_of_validations; x++)
		{
			real_fold_training_error[x] = 1000;
			real_fold_test_error[x] = 1000;
			real_fold_iteration_optimal[x] = -1;
			for(y=0; y<nb_of_iterations; y++)
				real_fold_validation_error[x][y] = 0;
		}
	}

	// compute the values of all the folds
	public void compute_global_real_fold_results()
	{
		int x;

		for(x=0; x<number_of_validations; x++)
			compute_real_fold_value(x);
	}

	// compute the values of a specific fold whose number is passed as parameter
	public void compute_real_fold_value(int fold_number)
	{
		int x,y;
		double temp_iteration_specific_participant;
		ValidationBoost temp_boost;
		double temporary_optimal;
		int temp_optimal_iteration;

		System.out.println("Fold :" + fold_number);
		temporary_optimal = 0;
		for (x=0;x<nb_of_participants;x++)
        {
			temp_boost = new ValidationBoost(val_participant_set[x][fold_number],nb_of_iterations,dimensionality);
			temp_boost.runValidation();
			for (y=0; y<nb_of_iterations; y++)
				real_fold_validation_error[fold_number][y] += temp_boost.validation_error[y];
			temp_iteration_specific_participant = temp_boost.optimal_nb_of_iterations;
			temporary_optimal += temp_iteration_specific_participant;
			//temporary_optimal += (temp_iteration_specific_participant/((double)Math.log(val_participant_set[x][fold_number].length)));
		}
		for (y=0; y<nb_of_iterations; y++)
			real_fold_validation_error[fold_number][y] = (real_fold_validation_error[fold_number][y]/nb_of_participants);
		save_fold_validation(print_fold_validation_error(fold_number),fold_number);
		//temporary_optimal = temporary_optimal * ((double)Math.log(nbdatapoints - val_test_set[fold_number].length));
		temp_optimal_iteration = Math.round(temporary_optimal/nb_of_participants);
		//temporary_optimal = (temporary_optimal/nb_of_participants) * ((double)Math.log(nbdatapoints - val_test_set[fold_number].length));
		//temp_optimal_iteration = Math.round(temporary_optimal);
		System.out.println("Optimal iteration : " + temp_optimal_iteration);
		real_fold_iteration_optimal[fold_number] = temp_optimal_iteration;
		real_fold_training_error[fold_number] = val_training_error[fold_number][temp_optimal_iteration];
		real_fold_test_error[fold_number] = val_test_error[fold_number][temp_optimal_iteration];
	}

	// construct the Alice, Bob and the test datasets for the current validation
	public void instance_particular_validation_set(int v)
	{
		int z;

		nbparticipantpoints = new int[nb_of_participants];
		for (z=0; z<nb_of_participants; z++)
			nbparticipantpoints[z] = val_participant_set[z][v].length;
		nbtestpoints = val_test_set[v].length;
		
		participant_set = new DataPoint[nb_of_participants][];
		for (z=0; z<nb_of_participants; z++)
			participant_set[z] = val_participant_set[z][v];
		test_set = val_test_set[v];
	}

	// save the results obtained by the current validation
	public void save_results_validation(int v)
	{
		int x,z;
		
		for (x=0; x<nb_of_iterations; x++)
		{
			val_training_error[v][x] = training_error[x];
			for (z=0; z<nb_of_participants; z++)
				val_participant_training_error[z][v][x] = participant_training_error[z][x];
			val_test_error[v][x] = test_error[x];
		}
	}
	
	// compute the global results of all the validations
	public void compute_global_validation_results()
	{
		int x,y,z;

		for (x=0; x< nb_of_iterations; x++)
		{
			training_error[x] = 0;
			for (z=0; z<nb_of_participants; z++)
				participant_training_error[z][x] = 0;
			test_error[x] = 0;
		}
		
		for (x=0; x< nb_of_iterations; x++)
		{
			for (y=0; y<number_of_validations; y++)
			{
				training_error[x] += val_training_error[y][x];
				for (z=0; z<nb_of_participants; z++)
					participant_training_error[z][x] += val_participant_training_error[z][y][x];
				test_error[x] += val_test_error[y][x];
			}
		}

		for (x=0; x< nb_of_iterations; x++)
		{
			training_error[x] = training_error[x]/number_of_validations;
			for (z=0; z<nb_of_participants; z++)
				participant_training_error[z][x] = participant_training_error[z][x]/number_of_validations;
			test_error[x] = test_error[x]/number_of_validations;
		}
	}
	
	// check if an integer is odd or even
	public boolean is_odd(int n)
	{
		boolean value_return;
		int temp_value;

		temp_value = n/2;
		if (n == (2* temp_value))
			value_return = false;
		else
			value_return = true;

		return value_return;
	}

	// check if the weights of the data points are normalized
	/*public void check_weight_normalization()
	{
		double sum;
		int x;
		
		sum = 0;
		for (x=0; x<nbalicepoints; x++)
			sum += alice_set[x].weight;
		for (x=0; x<nbbobpoints; x++)
			sum += bob_set[x].weight;
		//System.out.println("Somme des poids : " + sum);		
		
	}*/

	// compute and return the error of the MABoost classifier on a data set which is passed as parameter
	public double compute_training_error(DataPoint[] a, int nb_part, int num_iteration)
	{
		double value_return;
		double a_size;
		double temp_good_classification;
		double temp_bad_classification;
		double temp_value;
		double temp_class;
		int x,y;
				
		a_size = a.length;
		temp_good_classification = 0;
		temp_bad_classification = 0;
		for (x=0; x < a_size; x++)
		{
			temp_value = 0;
			// for each iteration compute the prediction of the weak classifier of this iteration
			// and keep track of the sum
			for (y=0; y < num_iteration; y++)
			{
				temp_class = common_weak_classifier_array[nb_part][y].classify_point(a[x]);
				temp_value += temp_class * common_weak_classifier_array[nb_part][y].weight_classifier;
			}
			// take the sign of the linear combination of the weak classifiers as the indicative function
			// for predicting the class of the currently examined datapoint
			if (temp_value < 0)
				temp_class = -1;
			else
				temp_class = 1;
			if (temp_class == a[x].point_class)
				temp_good_classification++;
			else
				temp_bad_classification++;
				
		}

		value_return = temp_bad_classification/a_size;		

		return value_return;
	}	

	// compute and return the error of the MABoost classifier on a data set which is passed as parameter
	public double compute_test_error(DataPoint[] a, int num_iteration)
	{
		double value_return;
		double a_size;
		double temp_good_classification;
		double temp_bad_classification;
		double temp_value, average_value;
		double temp_class;
		int x,y,z;
				
		a_size = a.length;
		average_value = 0;
		for (z=0; z<nb_of_participants; z++)
		{
			temp_good_classification = 0;
			temp_bad_classification = 0;
			for (x=0; x < a_size; x++)
			{
				temp_value = 0;
				// for each iteration compute the prediction of the weak classifier of this iteration
				// and keep track of the sum
				for (y=0; y < num_iteration; y++)
				{
					temp_class = common_weak_classifier_array[z][y].classify_point(a[x]);
					temp_value += temp_class * common_weak_classifier_array[z][y].weight_classifier;
				}
				// take the sign of the linear combination of the weak classifiers as the indicative function
				// for predicting the class of the currently examined datapoint
				if (temp_value < 0)
					temp_class = -1;
				else
					temp_class = 1;
				if (temp_class == a[x].point_class)
					temp_good_classification++;
				else
					temp_bad_classification++;
			}
			average_value += temp_bad_classification/a_size;
		}
		value_return = average_value/nb_of_participants;		

		return value_return;
	}	

	// classify a data point by using the combination of the weak classifiers of the MABoost classifier
	/*public int classify_point(DataPoint p)
	{
		int value_return;
		double temp_value;
		double temp_class;
		int x;
				

		temp_value = 0;
		// for each iteration compute the prediction of the weak classifier of this iteration
		// and keep track of the sum
		for (x=0; x < nb_of_iterations; x++)
		{
			temp_class = common_weak_classifier_array[x].classify_point(p);
			temp_value += temp_class * common_weak_classifier_array[x].weight_classifier;
		}
		// take the sign of the linear combination of the weak classifiers as the indicative function
		// for predicting the class of the currently examined datapoint
		if (temp_value < 0)
			value_return = -1;
		else
			value_return = 1;		

		return value_return;
	}*/	

	// use the final MABoost classifier computed to classify the test set
	/*public void classify_test_set()
	{
		int x;
		DataPoint current_point;
		int temp_class;

		nb_good_test = 0;
		nb_bad_test = 0;
		
		x = 0;
		while (x < nbtestpoints)
		{
      			current_point = test_set[x];
			temp_class = classify_point(current_point);

    			if (temp_class == current_point.point_class)
				nb_good_test++;
			else
				nb_bad_test++;
			x++;
		}
		
	}*/

	// use the final MABoost classifier computed to classify the training set which is composed of Alice and Bob dataset
	/*public void classify_training_set()
	{
		int x;
		DataPoint current_point;
		int temp_class;

		nb_good_training = 0;
		nb_bad_training = 0;
		
		x = 0;
		while (x < nbalicepoints)
		{
      			current_point = alice_set[x];
			temp_class = classify_point(current_point);

    			if (temp_class == current_point.point_class)
				nb_good_training++;
			else
				nb_bad_training++;
			x++;
		}
		x = 0;
		while (x < nbbobpoints)
		{
      			current_point = bob_set[x];
			temp_class = classify_point(current_point);

    			if (temp_class == current_point.point_class)
				nb_good_training++;
			else
				nb_bad_training++;
			x++;
		}
		
	}*/

	// compute the number of points in the main data set 
  	public void compute_data_size()
  	{
    		FileReader fr ;
    		BufferedReader stdin;
    		String str_temp;
    		String chunk_temp;
    		StringTokenizer str_tk;
		
		int temp_nb_points = 0;
		int temp_dimensionality = 0;
    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator");
    		//System.out.println("user path : " + user_path);

    		try
    		{
      			fr = new FileReader(user_path + data_filename);
      			stdin = new BufferedReader(fr);
      			str_temp = stdin.readLine();
      			temp_nb_points = 0;
			str_tk = new StringTokenizer(str_temp," ") ;
			temp_dimensionality = str_tk.countTokens() - 1;

			while (str_temp != null)
      			{
        			temp_nb_points++;
        			str_temp = stdin.readLine();
      			}
      			fr.close();
    		}
    		catch (Exception e)
    		{ System.out.println("Exception occured during the execution of compute_data_size : " + e);}
		
		dimensionality = temp_dimensionality;
		nbdatapoints = temp_nb_points;
		data_set = new DataPoint[nbdatapoints];
		

	  }

	// load the main dataset
  	public void load_dataset()
  	{
    		FileReader fr ;
    		BufferedReader stdin;
    		String str_temp;
    		String chunk_temp;
    		StringTokenizer str_tk;
    		double temp_double;
		Integer temp_int;
	
		int x,y;
		double[] temp_coordinates;
		String temp_str;

    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator");
    		//System.out.println("user path : " + user_path);
		
    		try
    		{
      			fr = new FileReader(user_path + data_filename);
      			stdin = new BufferedReader(fr);
      			str_temp = stdin.readLine();
      			y = 0;
			while (str_temp != null)
      			{
        			str_tk = new StringTokenizer(str_temp," ") ;
        			temp_coordinates = new double[dimensionality];

				x = 0;
				while (x < dimensionality)
				{
					chunk_temp = str_tk.nextToken();
        				temp_str = new String(chunk_temp);
					temp_double = new double(temp_str);
					temp_coordinates[x] = temp_double.doubleValue();
        				x++;
				}
				chunk_temp = str_tk.nextToken();
        			temp_str = new String(chunk_temp);
				temp_int = new Integer(temp_str);
				if (temp_int.intValue() == 0)
					temp_int = new Integer(-1);
				data_set[y] = new DataPoint(dimensionality,temp_coordinates,temp_int.intValue(),1);

        			str_temp = stdin.readLine();
				y++;
      			}
      			fr.close();
    		}
    		catch (Exception e)
    		{ System.out.println("Exception occured during the execution of load_data_set : " + e);}

	  }


	// save the representation of the MABoost Classifier
  	public void save_maboost_representation(String ma_rep)
  	{
    		FileWriter fw ;
    		BufferedWriter stdout;
		int str_length;
		
		str_length = ma_rep.length();

    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator");
    		
    		try
    		{
      			fw = new FileWriter(user_path + "stumps.txt");
      			stdout = new BufferedWriter(fw);
      			stdout.write(ma_rep,0,str_length);
			stdout.flush();
			stdout.close();
      			fw.close();
    		}
    		catch (Exception e)
    		{ System.out.println("Exception occured during the execution of save_maboost_representation : " + e);}

	  }

	// print the representation of the MABoost classifier
	/*public String print_maboost()
	{
		String value_return = new String("");
		int x;

		for (x=0; x < nb_of_iterations; x++)
		{	
			value_return = value_return + "alpha = " + common_weak_classifier_array[x].weight_classifier + " , ";
			value_return = value_return + "attribute = " + common_weak_classifier_array[x].rule_attribute_index + " : ";
			value_return = value_return + common_weak_classifier_array[x].lower_class + " < ";
			value_return = value_return + common_weak_classifier_array[x].lower_threshold + " <= ";
			value_return = value_return + common_weak_classifier_array[x].middle_class + " < ";
			value_return = value_return + common_weak_classifier_array[x].upper_threshold + " <= ";
			value_return = value_return + common_weak_classifier_array[x].upper_class + '\n';
		}

 		return value_return;
	}*/

	// save the overall training error of the MABoost classifier
  	public void save_training_error(String tr_error)
  	{
    		FileWriter fw ;
    		BufferedWriter stdout;
		int str_length;
		
		str_length = tr_error.length();

    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator");
    		
    		try
    		{
      			fw = new FileWriter(user_path + "training_error.txt");
      			stdout = new BufferedWriter(fw);
      			stdout.write(tr_error,0,str_length);
			stdout.flush();
			stdout.close();
      			fw.close();
    		}
    		catch (Exception e)
    		{ System.out.println("Exception occured during the execution of save_training_error : " + e);}

	  }

	// print the representation of the training error of the MABoost classifier
	public String print_training_error()
	{
		String value_return = new String("");
		int x;

		for (x=0; x < nb_of_iterations; x++)
			value_return = value_return + training_error[x] + '\n';

 		return value_return;
	}

	// save participant training error of the MABoost classifier
  	public void save_participant_error(String a_error, int num_p)
  	{
    		FileWriter fw ;
    		BufferedWriter stdout;
		int str_length;
		
		str_length = a_error.length();

    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator");
    		
    		try
    		{
      			fw = new FileWriter(user_path + "participant" + num_p + "_error.txt");
      			stdout = new BufferedWriter(fw);
      			stdout.write(a_error,0,str_length);
			stdout.flush();
			stdout.close();
      			fw.close();
    		}
    		catch (Exception e)
    		{ System.out.println("Exception occured during the execution of save_participant_training_error : " + e);}

	  }

	// print the representation of alice training error of the MABoost classifier
	public String print_participant_training_error(int num_p)
	{
		String value_return = new String("");
		int x;

		for (x=0; x < nb_of_iterations; x++)
			value_return = value_return + participant_training_error[num_p][x] + '\n';

 		return value_return;
	}

	// save the test error of the MABoost classifier
  	public void save_test_error(String te_error)
  	{
    		FileWriter fw ;
    		BufferedWriter stdout;
		int str_length;
		
		str_length = te_error.length();

    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator");
    		
    		try
    		{
      			fw = new FileWriter(user_path + "test_error.txt");
      			stdout = new BufferedWriter(fw);
      			stdout.write(te_error,0,str_length);
			stdout.flush();
			stdout.close();
      			fw.close();
    		}
    		catch (Exception e)
    		{ System.out.println("Exception occured during the execution of save_test_error : " + e);}

	  }

	// print the representation of the test error of the MABoost classifier
	public String print_test_error()
	{
		String value_return = new String("");
		int x;

		for (x=0; x < nb_of_iterations; x++)
			value_return = value_return + test_error[x] + '\n';

 		return value_return;
	}

	// print the data of the test set
	public void print_test_set()
	{
		int x;

		x = 0;
		while (x < nbtestpoints)
		{	
			System.out.println("Point number : " + x);
			test_set[x].printPoint();	
			x++;
		}
 	}
	
	// save the training error of the MABoost classifier for the validation whose number is passed as parameter
  	public void save_training_error_validation(String tr_error,int v)
  	{
    		FileWriter fw ;
    		BufferedWriter stdout;
		int str_length;
		
		str_length = tr_error.length();

    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator") + "training" + System.getProperty("file.separator");
    		
    		try
    		{
      			fw = new FileWriter(user_path + "training_error" + v + ".txt");
      			stdout = new BufferedWriter(fw);
      			stdout.write(tr_error,0,str_length);
			stdout.flush();
			stdout.close();
      			fw.close();
    		}
    		catch (Exception e)
    		{ System.out.println("Exception occured during the execution of save_training_error : " + e);}

	 }

	// save the test error of the MABoost classifier for the validation whose number is passed as parameter
  	public void save_test_error_validation(String te_error, int v)
  	{
    		FileWriter fw ;
    		BufferedWriter stdout;
		int str_length;
		
		str_length = te_error.length();

    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator") + "test" + System.getProperty("file.separator");
    		
    		try
    		{
      			fw = new FileWriter(user_path + "test_error" + v + ".txt");
      			stdout = new BufferedWriter(fw);
      			stdout.write(te_error,0,str_length);
			stdout.flush();
			stdout.close();
      			fw.close();
    		}
    		catch (Exception e)
    		{ System.out.println("Exception occured during the execution of save_test_error : " + e);}

	 }

	// save participant training error of the MABoost classifier for the validation whose number is passed as parameter
  	public void save_participant_error_validation(String a_error, int num_p, int v)
  	{
    		FileWriter fw ;
    		BufferedWriter stdout;
		int str_length;
		
		str_length = a_error.length();

    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator") + "participant"+ num_p + System.getProperty("file.separator");
    		
    		try
    		{
      			fw = new FileWriter(user_path + "participant_error" + v + ".txt");
      			stdout = new BufferedWriter(fw);
      			stdout.write(a_error,0,str_length);
			stdout.flush();
			stdout.close();
      			fw.close();
    		}
    		catch (Exception e)
    		{ System.out.println("Exception occured during the execution of save_participant_training_error : " + e);}

	  }

	// print the global results
	public String print_global_results()
	{
		String value_return = new String("");
		int x;
		double global_mean_training_error;
		double global_mean_test_error;
		double global_mean_iteration_optimal;
		double global_deviation_training_error;
		double global_deviation_test_error;
		double global_deviation_iteration_optimal;

		value_return += "Dataset : " + data_filename + '\n';
		value_return += "Number of data points : " + nbdatapoints + ", dimensionality : " + dimensionality + '\n';
		value_return += "Number of participants : " + nb_of_participants + '\n' + '\n' ;

		value_return += "Combined validation" + '\n';

		global_mean_training_error = 0;
		global_mean_test_error = 0;
		global_mean_iteration_optimal = 0;
		global_deviation_training_error = 0;
		global_deviation_test_error = 0;
		global_deviation_iteration_optimal = 0;

		for (x=0; x < number_of_validations; x++)
		{
			value_return += "fold " + x + ", training_error : " + real_fold_training_error[x] ;
			value_return += ", test error : " + real_fold_test_error[x] + ", num iteration : " + real_fold_iteration_optimal[x] + '\n';
			global_mean_training_error += real_fold_training_error[x];
			global_mean_test_error += real_fold_test_error[x];
			global_mean_iteration_optimal += real_fold_iteration_optimal[x];
		}
		global_mean_training_error = global_mean_training_error/number_of_validations;
		global_mean_test_error = global_mean_test_error/number_of_validations;
		global_mean_iteration_optimal = global_mean_iteration_optimal/number_of_validations;

		for (x=0; x < number_of_validations; x++)
		{
			global_deviation_training_error += (global_mean_training_error - real_fold_training_error[x]) * (global_mean_training_error - real_fold_training_error[x]);
			global_deviation_test_error += (global_mean_test_error - real_fold_test_error[x]) * (global_mean_test_error - real_fold_test_error[x]);
			global_deviation_iteration_optimal += (global_mean_iteration_optimal - real_fold_iteration_optimal[x]) * (global_mean_iteration_optimal - real_fold_iteration_optimal[x]);
		}
		global_deviation_training_error = global_deviation_training_error/number_of_validations;
		global_deviation_test_error = global_deviation_test_error/number_of_validations;
		global_deviation_iteration_optimal = global_deviation_iteration_optimal/number_of_validations;
		global_deviation_training_error = (double) Math.sqrt((double)global_deviation_training_error);
		global_deviation_test_error = (double) Math.sqrt((double)global_deviation_test_error);
		global_deviation_iteration_optimal = (double) Math.sqrt((double)global_deviation_iteration_optimal);

		value_return += '\n' + "Global training error : " + global_mean_training_error + " (" + global_deviation_training_error + ")" + '\n';
		value_return += "Global test error : " + global_mean_test_error + " (" + global_deviation_test_error + ")" + '\n';
		value_return += "Global number iteration optimal : " + global_mean_iteration_optimal + " (" + global_deviation_iteration_optimal + ")" + '\n';

		value_return += "Error after " + nb_of_iterations + " iterations" + '\n';

		global_mean_training_error = 0;
		global_mean_test_error = 0;
		global_mean_iteration_optimal = 0;
		global_deviation_training_error = 0;
		global_deviation_test_error = 0;
		global_deviation_iteration_optimal = 0;

		for (x=0; x < number_of_validations; x++)
		{
			value_return += "fold " + x + ", training_error : " + val_training_error[x][nb_of_iterations-1] ;
			value_return += ", test error : " + val_test_error[x][nb_of_iterations-1] + ", num iteration : " + nb_of_iterations + '\n';
			global_mean_training_error += val_training_error[x][nb_of_iterations-1];
			global_mean_test_error += val_test_error[x][nb_of_iterations-1];
			global_mean_iteration_optimal += nb_of_iterations;
		}
		global_mean_training_error = global_mean_training_error/number_of_validations;
		global_mean_test_error = global_mean_test_error/number_of_validations;
		global_mean_iteration_optimal = global_mean_iteration_optimal/number_of_validations;

		for (x=0; x < number_of_validations; x++)
		{
			global_deviation_training_error += (global_mean_training_error - val_training_error[x][nb_of_iterations-1]) * (global_mean_training_error - val_training_error[x][nb_of_iterations-1]);
			global_deviation_test_error += (global_mean_test_error - val_test_error[x][nb_of_iterations-1]) * (global_mean_test_error - val_test_error[x][nb_of_iterations-1]);
			global_deviation_iteration_optimal += (global_mean_iteration_optimal - nb_of_iterations) * (global_mean_iteration_optimal - nb_of_iterations);
		}
		global_deviation_training_error = global_deviation_training_error/number_of_validations;
		global_deviation_test_error = global_deviation_test_error/number_of_validations;
		global_deviation_iteration_optimal = global_deviation_iteration_optimal/number_of_validations;
		global_deviation_training_error = (double) Math.sqrt((double)global_deviation_training_error);
		global_deviation_test_error = (double) Math.sqrt((double)global_deviation_test_error);
		global_deviation_iteration_optimal = (double) Math.sqrt((double)global_deviation_iteration_optimal);

		value_return += '\n' + "Global training error : " + global_mean_training_error + " (" + global_deviation_training_error + ")" + '\n';
		value_return += "Global test error : " + global_mean_test_error + " (" + global_deviation_test_error + ")" + '\n';
		value_return += "Global number iteration optimal : " + global_mean_iteration_optimal + " (" + global_deviation_iteration_optimal + ")" + '\n';

		return value_return;
	}

	// save the global results
	public void save_global_results(String s_results)
	{
		FileWriter fw ;
		BufferedWriter stdout;
		int str_length;
		
		str_length = s_results.length();

		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator");
    		
		try
		{
			fw = new FileWriter(user_path + "global_results.txt");
			stdout = new BufferedWriter(fw);
			stdout.write(s_results,0,str_length);
			stdout.flush();
			stdout.close();
			fw.close();
		}
		catch (Exception e)
		{ System.out.println("Exception occured during the execution of save_global_results : " + e);}

	}

	// save the validation error of a particular fold
	public void save_fold_validation(String v_error, int fold_number)
	{
		FileWriter fw ;
		BufferedWriter stdout;
		int str_length;
		
		str_length = v_error.length();

		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator");
    		
		try
		{
			fw = new FileWriter(user_path + "validation_error_fold" + fold_number + ".txt");
			stdout = new BufferedWriter(fw);
			stdout.write(v_error,0,str_length);
			stdout.flush();
			stdout.close();
			fw.close();
		}
		catch (Exception e)
		{ System.out.println("Exception occured during the execution of save_fold_validation : " + e);}

	}

	// print the representation of the test error of the MABoost classifier
	public String print_fold_validation_error(int fold_number)
	{
		String value_return = new String("");
		int x;

		for (x=0; x < nb_of_iterations; x++)
			value_return = value_return + real_fold_validation_error[fold_number][x] + '\n';

		return value_return;
	}

}

		
		
