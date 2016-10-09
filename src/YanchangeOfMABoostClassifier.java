// MABoostClassifier.java
// manage the MABoost Classifier with decision stumps for 3 or more participants
// Sebastien Gambs

import java.util.* ;

import mloss.roc.Curve;

public class YanchangeOfMABoostClassifier
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
	public DataPointset[] participant_set_yan;
	public DataPoint[][] participant_set; // array containing the participants data set
	public int nb_good_training ; // number of points well classified on the training set
	public int nb_bad_training ; // number of points misclassified on the training set
	public int nb_good_test ; // number of points well classified on the training set
	public int nb_bad_test ; // number of points misclassified on the training set


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
	public YanchangeOfMABoostClassifier(arfffSplit[] inputdata, int participants, int numiter, int nth){
		nb_of_iterations=numiter;
		nb_of_participants=participants;
		nbparticipantpoints=new int[nb_of_participants];
		participant_set_yan=new DataPointset[nb_of_participants];
		for (int i=0; i<nb_of_participants;i++){
			double[][] temptrain=inputdata[i].getDoubleTrainCV(nth);
			nbparticipantpoints[i]=temptrain.length;
			if (i==0)
			    dimensionality=temptrain[0].length-1;
			participant_set_yan[i]=new DataPointset(temptrain);
		}
		
	}
	

	// contruct the MABoost Classifier
	public void constructMABoostClassifier()
	{
		double initial_weight;
		int x,z;
		int i;
		int temp_att, temp_class;

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
      				participant_set_yan[z].Pointset[x].weight = initial_weight;
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
				participant_weak_classifier_array[z][i].find_best_decision_stumps(participant_set_yan[z].Pointset);
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
					common_weak_classifier_array[z][i].compute_weighted_rates(participant_set_yan[z].Pointset);
					common_weak_classifier_array[z][i].compute_penalties();
					common_weak_classifier_array[z][i].compute_weight_classifier();
				}
				// update the weights of the participant points
				for (z=0; z<nb_of_participants; z++)
				{
					for (x=0; x < nbparticipantpoints[z]; x++)
					{
      						temp_class = common_weak_classifier_array[z][i].classify_point(participant_set_yan[z].Pointset[x]);
						//if (temp_class == 0)
						//	participant_set[z][x].weight = participant_set[z][x].weight * common_weak_classifier_array[z][i].penalty_abstention;
						//else
						//{	
							if (participant_set_yan[z].Pointset[x].point_class == temp_class)
								participant_set_yan[z].Pointset[x].weight = participant_set_yan[z].Pointset[x].weight * common_weak_classifier_array[z][i].penalty_good_classification;
							else
								participant_set_yan[z].Pointset[x].weight = participant_set_yan[z].Pointset[x].weight * common_weak_classifier_array[z][i].penalty_misclassification;
						//}
					}
				}
				// compute and store all the errors (training and test)
				//temp_weighted_abstention = common_weak_classifier_array[i].weighted_abstention_rate;
				//temp_weighted_error = common_weak_classifier_array[i].weighted_error;
				//temp_weighted_good = common_weak_classifier_array[i].weighted_good_rate;
				i++;
			}
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


	// compute and return the error of the MABoost classifier on a data set which is passed as parameter
	public double[] compute_test_error(DataPoint[] a, int num_iteration)
	{
		double value_return[]=new double[6];
		int a_size;
		double temp_good_classification;
		double temp_bad_classification;
		double temp_value, average_value;
		double temp_class;
		int x,y,z;
				
		a_size = a.length;
		double predict[]=new double[a_size];
		int actralllable[]=new int[a_size];
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
				if (x==0){
				predict[x]=temp_value;
				actralllable[x]=a[x].point_class;
				}
				else predict[x] +=temp_value;
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
			average_value += temp_good_classification/a_size;
		}
		
		 Curve rocAnalysisA = new Curve.PrimitivesBuilder()
         .predicteds(predict)
         .actuals(actralllable)
         .build();
		 
        value_return[0] = average_value/nb_of_participants;		
        value_return[1]=rocAnalysisA.rocArea();
        value_return[2]=rocAnalysisA.convexHull().rocArea();
        value_return[3]=rocAnalysisA.bestfmeasure()[0];
        value_return[4]=rocAnalysisA.bestfmeasure()[1];
        value_return[5]=rocAnalysisA.bestfmeasure()[2];
        
		return value_return;
	}	

	
}

		
		
