

import java.io.* ;
import java.util.* ;

public class ValidationBoost
{
	public int dimensionality ; // dimensionality of the space
	public int max_nb_of_iterations ; // number of iterations of the boosting process
	public int optimal_nb_of_iterations ; // optimal number of iterations returned by the class
	public int nbdatapoints ; // total number of data points
	public int nbtrainingpoints ; // number of points in the training set
	public int nbvalidationpoints ; // number of points in the validation set
	public AdaBasicClassifier[] common_weak_classifier_array ; // common weak classifiers stored in an array
	public AdaWeakClassifier[] training_weak_classifier_array; // weak classifiers stored in an array
	public double[] training_error; // array storing the overall training error of the MABoost classifier at each iteration
	public double[] validation_error; // array storing the validation error of the MABoost classifier at each iteration

	public DataPoint[] data_set ; // array containing the points of the data set
	public DataPoint[] training_set ; // array containing the points of the training set
	public DataPoint[] validation_set ; // array containing the points of the test set

	public int number_of_validations; // number of validations used
	public double[][] val_training_error; // array storing the overall training error of the MABoost classifier at each iteration and each validation
	public double[][] val_validation_error; // array storing the validation error of the MABoost classifier at each iteration and each validation
	public DataPoint[][] val_training_set ; // array containing the points of the test set for each validation
	public DataPoint[][] val_validation_set; // array containing the dataset of Alice for each validation

	// constructor
	public ValidationBoost(DataPoint[] array_original, int max_iterations, int d)
	{
		int x;

		nbdatapoints = array_original.length;
		dimensionality = d;
		// initialize a temporary array
		data_set = new DataPoint[nbdatapoints]; 
		for (x=0; x < nbdatapoints; x++)
			data_set[x] = array_original[x].cloneDataPoint();

		optimal_nb_of_iterations = 0;
		max_nb_of_iterations = max_iterations;
		number_of_validations = 2;
	}

	// run the AdaBoost validation
	public void runValidation()
	{
		String adaboost_representation;
		String training_error_representation;
		String validation_error_representation;
		DataPoint[] temp_data_array;
		int x,y;

		// compute the size of the dataset, load the dataset and construct the training and the test set from this dataset
		//compute_data_size();
		//load_dataset();

		// construct the AdaBoost classifier based on the training set
		// construct_sets();
		// constructAdaBoostClassifier();
		//adaboost_representation = print_adaboost();
		//System.out.print(adaboost_representation);
		
		construct_validation_sets();
		for (x=0; x < number_of_validations; x++)
		{
			System.out.println("Validation number : "+ (x+1));
			instance_particular_validation_set(x);
			constructAdaBoostClassifier();
			training_error_representation = print_training_error();
			validation_error_representation = print_validation_error();
			save_training_error_validation(training_error_representation,x);
			save_validation_error(validation_error_representation,x);
			save_results_validation(x);
		}
		compute_global_validation_results();

		// save the training error and the test error on files
		// respectively in "training_error.txt" and "test_error.txt"
		//training_error_representation = print_training_error();
		//test_error_representation = print_test_error();
		//save_training_error(training_error_representation);
		//save_test_error(test_error_representation);
		//save_paboost_representation(paboost_representation);

		//classify_training_set();
		//System.out.println("Number of points well classified in the training set : " + nb_good_training);
		//System.out.println("Number of points misclassified in the training set : " + nb_bad_training);
		//classify_test_set();
		//System.out.println("Number of points well classified in the test set : " + nb_good_test);
		//System.out.println("Number of points misclassified in the test set : " + nb_bad_test);
		//System.out.println(nb_bad_training + " " + nb_bad_test);
		//print_training_set();
		//print_test_set();
	}

	// contruct the AdaBoost Classifier
	public void constructAdaBoostClassifier()
	{
		double initial_weight;
		int x,y;
		int i;
		int temp_att, temp_class;
		DataPoint[] temp_partition_set;
		double temp_weighted_abstention, temp_weighted_error, temp_weighted_good;

		// initialisation of the variables of the PABoost classifier
		common_weak_classifier_array = new AdaBasicClassifier[max_nb_of_iterations];
		training_weak_classifier_array = new AdaWeakClassifier[max_nb_of_iterations];
		training_error = new double[max_nb_of_iterations];
		validation_error = new double[max_nb_of_iterations];

		// the initial weight of all the datapoints is set to 1 divided by the number of datapoints to indicate
		// that a priori all the datapoints have the same difficulty 
		initial_weight = 1/((double) (nbtrainingpoints));
		for (x=0; x < nbtrainingpoints; x++)
      			training_set[x].weight = initial_weight;
		temp_partition_set = new DataPoint[nbtrainingpoints];

		i = 0;
		while (i < max_nb_of_iterations)
		{
      			//System.out.println("Iteration number : " + i);
			//System.out.println("Number of iterations : " + nb_of_iterations + " dimensionality : " + dimensionality);
			//check_weight_normalization();
			for (x=0; x<nbtrainingpoints; x++)
				temp_partition_set[x] = training_set[x].cloneDataPoint();

			// find the best decision stumps regarding each dimension
			training_weak_classifier_array[i] = new AdaWeakClassifier(dimensionality);
			training_weak_classifier_array[i].find_best_decision_stumps(temp_partition_set);
			// construct the merged abstention classifier from Alice and Bob weak classifiers
			common_weak_classifier_array[i] = new AdaBasicClassifier(dimensionality,training_weak_classifier_array[i]);
			// compute the weighted rates, the penalties and the weight of the merged classifier 
			common_weak_classifier_array[i].compute_weighted_rates(temp_partition_set);
			common_weak_classifier_array[i].compute_penalties();
			common_weak_classifier_array[i].compute_weight_classifier();
			// update the weights of the training set points
			for (x=0; x < nbtrainingpoints; x++)
			{
      				temp_class = common_weak_classifier_array[i].classify_point(training_set[x]);
				if (training_set[x].point_class == temp_class)
					training_set[x].weight = training_set[x].weight * common_weak_classifier_array[i].penalty_good_classification;
				else
					training_set[x].weight = training_set[x].weight * common_weak_classifier_array[i].penalty_misclassification;	
			}
			// compute and store all the errors (training and test)
			i++;
			training_error[i-1] = compute_error(training_set,i);
			validation_error[i-1] = compute_error(validation_set,i);
		}	
	}

	// construct the training set and the validation set for each validation
	public void construct_validation_sets()
	{
		DataPoint[] temp_data_array;
		int[] size_block_data;
		double temp_block_size;
		double last_block_size;
		int temp_validation_size;
		int temp_training_size;
		int x,y;
		int current_position;
		int a,b,t;
	
		// initialize the array that will contain the different errors for each validation
		val_training_error = new double[number_of_validations][max_nb_of_iterations];
		val_validation_error = new double[number_of_validations][max_nb_of_iterations];

		// initialize a temporary array
		temp_data_array = new DataPoint[nbdatapoints]; 
		for (x=0; x < nbdatapoints; x++)
      			temp_data_array[x] = data_set[x].cloneDataPoint();
		
		size_block_data = new int[number_of_validations];
		
		// initialize the array which will contains the training and the validation dataset during the different validation
		val_training_set = new DataPoint[number_of_validations][];
		val_validation_set = new DataPoint[number_of_validations][];

		// compute the size of the different blocks
		temp_block_size = Math.round(((double)nbdatapoints)/number_of_validations);
		last_block_size = (double) nbdatapoints - (temp_block_size * (number_of_validations-1));
		for (x = 0; x<number_of_validations-1; x++)
			size_block_data[x] = (int) temp_block_size;
		size_block_data[number_of_validations-1] = (int) last_block_size;
		
		// construct the training and the validation dataset for each validation step
		for (x = 0; x<number_of_validations; x++)
		{
			temp_training_size = 0;
			temp_validation_size = 0;
			// compute the size of the training and the test dataset for the current validation
			for (y = 0; y <number_of_validations; y++)
			{
				if (y == x)
					temp_validation_size = size_block_data[y];
				else
					temp_training_size += size_block_data[y];
			}
			
			val_training_set[x] = new DataPoint[temp_training_size];
			val_validation_set[x] = new DataPoint[temp_validation_size];
	
            current_position = 0;
			a = 0;
			for (y = 0; y <number_of_validations; y++)
			{
				// current block is used as the test dataset 
				if (y == x)
				{	
					t=0;
					while(t<size_block_data[y])
					{
						val_validation_set[x][t] = temp_data_array[current_position].cloneDataPoint();
						t++;
						current_position++;
					}
				}
				else
				{
					t=0;
					while(t<size_block_data[y])
					{
						val_training_set[x][a] = temp_data_array[current_position].cloneDataPoint();
						t++;
						a++;
						current_position++;
					}
				}
			}
		}
		
	}

	// construct the training and the validation datasets for the current validation
	public void instance_particular_validation_set(int v)
	{
		nbtrainingpoints = val_training_set[v].length;
		nbvalidationpoints = val_validation_set[v].length;
		training_set = val_training_set[v];
		validation_set = val_validation_set[v];
	}

	// save the results obtained by the current validation
	public void save_results_validation(int v)
	{
		int x;
		
		for (x=0; x<max_nb_of_iterations; x++)
		{
			val_training_error[v][x] = training_error[x];
			val_validation_error[v][x] = validation_error[x];
		}
	}
	
	// compute the global results of all the validations
	public void compute_global_validation_results()
	{
		int x,y;
        int num_optimal_iteration;
		double value_error_optimal_iteration;
		double temp_optimal_number_of_iterations;

		for (x=0; x< max_nb_of_iterations; x++)
		{
			training_error[x] = 0;
			validation_error[x] = 0;
		}
		
		for (x=0; x< max_nb_of_iterations; x++)
		{
			for (y=0; y<number_of_validations; y++)
			{
				training_error[x] += val_training_error[y][x];
				validation_error[x] += val_validation_error[y][x];
			}
		}

		for (x=0; x< max_nb_of_iterations; x++)
		{
			training_error[x] = training_error[x]/number_of_validations;
			validation_error[x] = validation_error[x]/number_of_validations;
		}

		temp_optimal_number_of_iterations = 0;
		//for (x=0; x< number_of_validations; x++)
		//{
			num_optimal_iteration = 0;
			value_error_optimal_iteration = 1000;
			for (y=0; y< max_nb_of_iterations; y++)
			{
				//if (val_validation_error[x][y] < value_error_optimal_iteration)
				if (validation_error[y] < value_error_optimal_iteration)
				{
					num_optimal_iteration = y;
					value_error_optimal_iteration = validation_error[y];
					//value_error_optimal_iteration = val_validation_error[x][y];
				}
			}	
            //temp_optimal_number_of_iterations += num_optimal_iteration;
            temp_optimal_number_of_iterations = num_optimal_iteration;
		//}
		//temp_optimal_number_of_iterations = temp_optimal_number_of_iterations/number_of_validations;
		optimal_nb_of_iterations = (int) Math.round(temp_optimal_number_of_iterations);
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
	public void check_weight_normalization()
	{
		double sum;
		int x;
		
		sum = 0;
		for (x=0; x<nbtrainingpoints; x++)
			sum += training_set[x].weight;

		System.out.println("Somme des poids : " + sum);		
		
	}

	// compute and return the error of the AdaBoost classifier on a data set which is passed as parameter
	public double compute_error(DataPoint[] a, int num_iteration)
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
				temp_class = common_weak_classifier_array[y].classify_point(a[x]);
				temp_value += temp_class * common_weak_classifier_array[y].weight_classifier;
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

	// classify a data point by using the combination of the weak classifiers of the AdaBoost classifier
	public int classify_point(DataPoint p)
	{
		int value_return;
		double temp_value;
		double temp_class;
		int x;
				

		temp_value = 0;
		// for each iteration compute the prediction of the weak classifier of this iteration
		// and keep track of the sum
		for (x=0; x < max_nb_of_iterations; x++)
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
	}	

	// print the representation of the training error of the AdaBoost classifier
	public String print_training_error()
	{
		String value_return = new String("");
		int x;

		for (x=0; x < max_nb_of_iterations; x++)
			value_return = value_return + training_error[x] + '\n';

 		return value_return;
	}

	// print the representation of the test error of the AdaBoost classifier
	public String print_validation_error()
	{
		String value_return = new String("");
		int x;

		for (x=0; x < max_nb_of_iterations; x++)
			value_return = value_return + validation_error[x] + '\n';

 		return value_return;
	}

	
	// print the data of the test set
	/*public void print_test_set()
	{
		int x;

		x = 0;
		while (x < nbtestpoints)
		{	
			System.out.println("Point number : " + x);
			test_set[x].printPoint();	
			x++;
		}
 	}*/
	
	// save the training error of the AdaBoost classifier for the validation whose number is passed as parameter
  	public void save_training_error_validation(String tr_error,int v)
  	{
    		FileWriter fw ;
    		BufferedWriter stdout;
		int str_length;
		
		str_length = tr_error.length();

    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator") + "validation" + System.getProperty("file.separator");
    		
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

	// save the test error of the AdaBoost classifier for the validation whose number is passed as parameter
  	public void save_validation_error(String te_error, int v)
  	{
    		FileWriter fw ;
    		BufferedWriter stdout;
		int str_length;
		
		str_length = te_error.length();

    		String user_path = System.getProperty("user.dir") + System.getProperty("file.separator") + "validation" + System.getProperty("file.separator");
    		
    		try
    		{
      			fw = new FileWriter(user_path + "validation_error" + v + ".txt");
      			stdout = new BufferedWriter(fw);
      			stdout.write(te_error,0,str_length);
			stdout.flush();
			stdout.close();
      			fw.close();
    		}
    		catch (Exception e)
    		{ System.out.println("Exception occured during the execution of save_validation_error : " + e);}

	 }

	
}

		
		
