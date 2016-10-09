// MABoostMain.java
// contains the class used to launch the MABoost algorithm
// Sebastien Gambs

import java.io.* ;

public class MABoostMain
{

	// constructor
	public MABoostMain()
	{
	}

	// main function
	public static void main (String[] args)
	{
		String data_filename;
 		int nb_of_iterations;
		int nb_of_participants;
		MABoostClassifier mac;
		Integer temp_integer;

		if (args.length != 3)
		{
			System.out.println("The number of arguments of a call to MABoostMain is 3:");
			System.out.println("java AdaboostMain data_file nb_of_participants nb_of_iterations");
		}
		else
		{
			data_filename = new String(args[0]);
			temp_integer = new Integer(args[1]);
			nb_of_participants = temp_integer.intValue();
			temp_integer = new Integer(args[2]);
			nb_of_iterations = temp_integer.intValue();
			mac = new MABoostClassifier(data_filename,nb_of_iterations,nb_of_participants);
			mac.runClassification();
		}
 
	}
}

		
		
