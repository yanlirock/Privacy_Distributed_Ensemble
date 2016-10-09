// DataPoint.java
// holds the information regarding a data point
// Sebastien Gambs

import java.io.* ;

public class DataPoint
{
	public int dimensionality ; // dimensionality of the space in which the point lives
	public double[] coordinates ; // coordinates of the point
	public int point_class; // class of the DataPoint (in this case either +1 or -1)
 	public double weight; // current weight of the DataPoint

	// constructor
	public DataPoint(int d,double[] tab_c,int c, double w)
	{
		int x;

		dimensionality = d;
		coordinates = new double[d];
		for (x=0 ; x<d ; x++)
			coordinates[x] = tab_c[x];
		point_class = c;
		weight = w;
	}

	// return a clone of the current data point
	public DataPoint cloneDataPoint()
	{
		DataPoint new_data_point;
	
		new_data_point = new DataPoint(dimensionality,coordinates,point_class,weight);
		return new_data_point;
	}

	// print the data of the points
	public void printPoint()
	{
		int x;

		System.out.println("Dimensionality : " + dimensionality);
		x = 0;
		while (x < dimensionality)
		{
			System.out.println(coordinates[x]);
			x++;
		}
		System.out.println("Class: " + point_class);
		System.out.println("Weight: " + weight);		
	}
}

		
		
