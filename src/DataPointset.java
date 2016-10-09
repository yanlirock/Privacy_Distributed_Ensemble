public class DataPointset {
	private double[][] data;
	private int numpoint;
	private int dimensionality;
	public DataPoint[] Pointset;
	
	public DataPointset(double[][] indata){
		data=indata;
		numpoint=indata.length;
		dimensionality=data[0].length-1;
		Pointset=new DataPoint[numpoint];
		for(int j=0; j<numpoint; j++){
			int lable;
			if (data[j][dimensionality]==0)
				lable=-1;
			else lable=1;
			Pointset[j]=new DataPoint(dimensionality,data[j],lable, (double)(1/numpoint));
		}
	}

}
