

import java.io.File;
import java.text.DecimalFormat;

import mloss.roc.Curve;
import weka.core.Instances;


public class RunmultBoostEvaluateforMultiinput {
	
	
    public static void main(String[] args) {
        if (args.length != 4) {
            String errorReport = "DynamicParaBoostEvaluate: the Correct arguments are \n"
                            + "java DynamicParaBoostEvaluate.RunDynamicParaBoostEvaluate "
                            + "<num boost iterations> <num Folds> <data directory> <num files>";
            System.out.println(errorReport);
            System.exit(0);
	    }
	    int ni = Integer.parseInt(args[0]);
	    int numFolds = Integer.parseInt(args[1]);
	    final File folder = new File(args[2]);
        int numAgents = Integer.parseInt(args[3]);

        DecimalFormat FourPlaces = new DecimalFormat("#0.0000");


        // this is for all test data
        double errorRate[] = new double[numFolds];
        StatCalc sc = new StatCalc();
        StatCalc auc =new StatCalc();
        StatCalc Fmeasure= new StatCalc();
        StatCalc recall= new StatCalc();
        StatCalc precision= new StatCalc();
        StatCalc Maxauc= new StatCalc();
        // this is for local node test data in integrated model
        double errorRaten[][] = new double[numAgents][numFolds];
        StatCalc scn[] = new StatCalc[numAgents];
        StatCalc aucn[] =new StatCalc[numAgents];
        StatCalc Fmeasuren[] = new StatCalc[numAgents];
        StatCalc recalln[]= new StatCalc[numAgents];
        StatCalc precisionn[] = new StatCalc[numAgents];
        StatCalc Maxaucn[]= new StatCalc[numAgents];
        for(int ini=0; ini<numAgents;ini++){
            scn[ini] = new StatCalc();
            aucn[ini] =new StatCalc();
            Fmeasuren[ini] = new StatCalc();
            recalln[ini]= new StatCalc();
            precisionn[ini] = new StatCalc();
            Maxaucn[ini]= new StatCalc();
        }
        // this is for local node test data in local model
        double errorRatel[][] = new double[numAgents][numFolds];
        StatCalc scl[] = new StatCalc[numAgents];
        StatCalc aucl[] =new StatCalc[numAgents];
        StatCalc Fmeasurel[] = new StatCalc[numAgents];
        StatCalc recalll[]= new StatCalc[numAgents];
        StatCalc precisionl[] = new StatCalc[numAgents];
        StatCalc Maxaucl[]= new StatCalc[numAgents];
        for(int ini=0; ini<numAgents;ini++){
            scl[ini] = new StatCalc();
            aucl[ini] =new StatCalc();
            Fmeasurel[ini] = new StatCalc();
            recalll[ini]= new StatCalc();
            precisionl[ini] = new StatCalc();
            Maxaucl[ini]= new StatCalc();
        }
        
        arfffSplit afsarray[] =new arfffSplit[numAgents];
        //get Multiple input source
        int kj=0;
        int total=0;
        int index=0;
        int sizeofAgents[]= new int[numAgents];
        for (final File fileEntry : folder.listFiles())
		{
			String arfffname = fileEntry.getName();
		    arfffname = args[2] + "/" + arfffname;
			afsarray[kj] = new arfffSplit(arfffname, numFolds);
			afsarray[kj].getArffData();
			afsarray[kj].stratifiedcvpre();
			int tepnumber=afsarray[kj].getObservationNumber();
			sizeofAgents[kj]=tepnumber;
			total=total+tepnumber;
			kj=kj+1;
		}
        //get the total number of data
        double[] allpredictLabels=new double[total];
        int[] allactualLabels=new int[total];
        
        
        //doing 10cv here
        for(int i=0;i<numFolds;i++){
            System.out.println("Building Model for fold "+(i+1));            
            nodetest nodedata[]= new nodetest[numAgents];
        	YanchangeOfMABoostClassifier adb=new YanchangeOfMABoostClassifier(afsarray, numAgents, ni, i);
            adb.constructMABoostClassifier();
            //for each local source get test data
            DataPoint test[]=null;
            for (int j=0;j<numAgents;j++)
            {
	                double[][] testData=afsarray[j].getDoubleTestCV(i);
	            int dd=testData[0].length-1;
	            for (int t=0;t<testData.length;t++)
	            {
	            	int lable;
					if (testData[t][testData.length-1]==0)
						lable=-1;
					else lable=1;
					test[t]=new DataPoint(dd,testData[t],lable, (double)(1/testData.length));
				}
	            double[] res=adb.compute_test_error(test, ni);
                scl[j].enter(res[0]);
				aucl[j].enter(res[1]);
				Maxaucl[j].enter(res[2]);
				Fmeasurel[j].enter(res[3]);
	            recalll[j].enter(res[4]);
	            precisionl[j].enter(res[5]);
            }
        }
        //end of the 10cv
        
        //print the evaluation of the integrated model on local data
        System.out.println("Evaluate the integrated model on each node");
        System.out.println("Accuracy auc AUC Fmeasure Recall Precision");
        for(int pi=0;pi<numAgents;pi++){
        	System.out.println(FourPlaces.format(1-scn[pi].getMean())+"("+FourPlaces.format(scn[pi].getStandardDeviation())+") "
                    +FourPlaces.format(aucn[pi].getMean())+"("+FourPlaces.format(aucn[pi].getStandardDeviation())+") "
                    +FourPlaces.format(Maxaucn[pi].getMean())+"("+FourPlaces.format(Maxaucn[pi].getStandardDeviation())+") "
                    +FourPlaces.format(Fmeasuren[pi].getMean())+"("+FourPlaces.format(Fmeasuren[pi].getStandardDeviation())+") "
                    +FourPlaces.format(recalln[pi].getMean())+"("+FourPlaces.format(recalln[pi].getStandardDeviation())+") "
                    +FourPlaces.format(precisionn[pi].getMean())+"("+FourPlaces.format(precisionn[pi].getStandardDeviation())+") "
                    );
        }
        
        
        //print the evaluation of the local model on local data
        System.out.println("Evaluate the local model on each node");
        System.out.println("Accuracy auc AUC Fmeasure Recall Precision");
        for(int pi=0;pi<numAgents;pi++){
        	System.out.println(FourPlaces.format(1-scl[pi].getMean())+"("+FourPlaces.format(scl[pi].getStandardDeviation())+") "
                    +FourPlaces.format(aucl[pi].getMean())+"("+FourPlaces.format(aucl[pi].getStandardDeviation())+") "
                    +FourPlaces.format(Maxaucl[pi].getMean())+"("+FourPlaces.format(Maxaucl[pi].getStandardDeviation())+") "
                    +FourPlaces.format(Fmeasurel[pi].getMean())+"("+FourPlaces.format(Fmeasurel[pi].getStandardDeviation())+") "
                    +FourPlaces.format(recalll[pi].getMean())+"("+FourPlaces.format(recalll[pi].getStandardDeviation())+") "
                    +FourPlaces.format(precisionl[pi].getMean())+"("+FourPlaces.format(precisionl[pi].getStandardDeviation())+") "
                    );
        }
        
        
        //this is the mean and std of 10cv
        System.out.println("The average accuracy of the Predict model for all 10 fold is: "+FourPlaces.format(1-sc.getMean())+", with the standard deviation ("
                +FourPlaces.format(sc.getStandardDeviation())+")");
        System.out.println("The average auc of the Predict model for all 10 fold is: "+FourPlaces.format(auc.getMean())+", with the standard deviation ("
                +FourPlaces.format(auc.getStandardDeviation())+")");
        System.out.println("The average AUC of the Predict model for all 10 fold is: "+FourPlaces.format(Maxauc.getMean())+", with the standard deviation ("
                +FourPlaces.format(Maxauc.getStandardDeviation())+")");
        System.out.println("The average F_measure of the Predict model for all 10 fold is: "+FourPlaces.format(Fmeasure.getMean())+", with the standard deviation ("
                +FourPlaces.format(Fmeasure.getStandardDeviation())+")");
        System.out.println("The average recall of the Predict model for all 10 fold is: "+FourPlaces.format(recall.getMean())+", with the standard deviation ("
                +FourPlaces.format(recall.getStandardDeviation())+")");
        System.out.println("The average precision of the Predict model for all 10 fold is: "+FourPlaces.format(precision.getMean())+", with the standard deviation ("
                +FourPlaces.format(precision.getStandardDeviation())+")");
        
        // this is whole AUC
		 Curve rocallAnalysis = new Curve.PrimitivesBuilder()
         .predicteds(allpredictLabels)
         .actuals(allactualLabels)
         .build();
		 Curve convexHull = rocallAnalysis.convexHull();
	    double maxArea = convexHull.rocArea();
	    // Get the points for later plotting
	    double[][] rocPoints = convexHull.rocPoints();
		/*
		 * Print the generated fit
		 */
		System.out.println("The AUC of the Prediction Model for all is: "+FourPlaces.format(rocallAnalysis.rocArea()));
		System.out.println("The maxAUC of the Prediction Model for all is: "+FourPlaces.format(maxArea));
		double[] Fm=rocallAnalysis.bestfmeasure();
		System.out.println("The F_measure of the Prediction Model for all is: "+FourPlaces.format(Fm[0])+", with the corresponding recall ("+FourPlaces.format(Fm[1])+") and precision ("+FourPlaces.format(Fm[2])+")");	
    }
    
    //this is the end of main
    
    //this function is used to change instance data to double matrix
    public static double[][] instoDouble(Instances ins){
        int numIns = ins.numInstances();
        int numAtt = ins.numAttributes();
        double splitData[][] = new double[numIns][numAtt];
        double tmp[] = new double[numIns];
        for(int k=0; k<splitData[0].length; k++){
            tmp = ins.attributeToDoubleArray(k);
            for(int l=0; l<splitData.length; l++){
                splitData[l][k] = tmp[l];
            }
        }
        return splitData;
    }
    
    /*
     * this function is used to combine two matrix in matlab it's [X1;X2]
     * */
    public static double[][] MatrixColCombine(double[][] matrix1, double[][] matrix2){
    	int numrow1=matrix1.length;
    	int numrow2=matrix2.length;
    	int nummcol=matrix1[0].length;
    	if (matrix1[0].length != matrix2[0].length){
    	throw new IllegalArgumentException(
    	        "Two matrix must have same number of column");
    	    }
    	double[][] Combine=new double[numrow1+numrow2][nummcol];
    	int currentrow=0;
    	for(int i=0;i<numrow1;i++){
    		for(int j=0;j<nummcol;j++){
    			Combine[currentrow][j]=matrix1[i][j];
    			}
    		currentrow +=1;
    	}
    	for(int i=0;i<numrow2;i++){
    		for(int j=0;j<nummcol;j++){
    			Combine[currentrow][j]=matrix2[i][j];
    		}
    		currentrow +=1;
    	}
    	return Combine;
    }
}