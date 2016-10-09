


import java.io.File;
import weka.core.converters.ArffLoader;
import weka.core.Instances;
import java.io.IOException;
import java.util.Random;

public class arfffSplit {

    private String arfffname;
    private int numSplit;
    private Instances data;
    private int orgnumAtt;
    private int numAtt;
    public int selectedindex[];

    public arfffSplit(String f, int num){
        arfffname = f;
        numSplit = num;      
    }

    public void getArffData(){
        try{
            File file = new File(arfffname);
            ArffLoader arff = new ArffLoader();
            arff.setSource(file);
            data = arff.getDataSet();
            orgnumAtt = data.numAttributes();
            int[] numvalues=new int[orgnumAtt];
            for (int num=0;num<orgnumAtt-1;num++)
            	numvalues[num]=data.numDistinctValues(num);
            /*
            int iik=0;
            for (int ii=1; ii<orgnumAtt-1;ii++){
            	iik=iik+1;
            	if (numvalues[ii]==1){
            	data.deleteAttributeAt(iik);
            	iik=iik-1;
            	}
            }
            */
            numAtt = data.numAttributes();
            selectedindex=new int[numAtt-1];
            int index=0;
            for (int num2=0;num2<orgnumAtt-1;num2++){
            	if (numvalues[num2]!=1){
            		selectedindex[index]=num2;
            		index=index+1;
            	}
            }		
            // setting class attribute if the data format does not provide this information
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception e){
            new IOException("Some IO problem here!!!!");
        }
    }
    
    /*this function is used to prepare stratified cross-validation  
     * */
    public void stratifiedcvpre(){
        data.randomize(new Random(1));
        if (data.classAttribute().isNominal())
            data.stratify(numSplit);
    }
    
    public double[][] getDoubleTestCV(int i){
        Instances split = data.testCV(numSplit, i);
        int numIns = split.numInstances();
        double splitData[][] = new double[numIns][numAtt];
        double tmp[] = new double[numIns];

        for(int j=0; j<splitData[0].length; j++){
            tmp = split.attributeToDoubleArray(j);
            for(int k=0; k<splitData.length; k++){
                splitData[k][j] = tmp[k];
            }
        }
        return splitData;
    }
    public double[][] getDoubleTrainCV(int i){
        Instances split = data.trainCV(numSplit, i);
        int numIns = split.numInstances();
        double splitData[][] = new double[numIns][numAtt];
        double tmp[] = new double[numIns];

        for(int j=0; j<splitData[0].length; j++){
            tmp = split.attributeToDoubleArray(j);
            for(int k=0; k<splitData.length; k++){
                splitData[k][j] = tmp[k];
            }
        }
        return splitData;
    }
    public double[][] getDoubleVerifyCV(int i, double rate){
        Instances split = data.trainCV(numSplit, i);
        int numIns = split.numInstances();
        split.randomize(new Random(1));
        double num=numIns*rate;
        double splitData[][] = new double[(int)num][numAtt];
        double tmp[] = new double[numIns];
        for(int j=0; j<splitData[0].length; j++){
            tmp = split.attributeToDoubleArray(j);
            for(int k=0; k<splitData.length; k++){
                splitData[k][j] = tmp[k];
            }
        }
        return splitData;
    }
    public double[][] getDoubleresampleCV(int i){
        Instances splito = data.trainCV(numSplit, i);
        int numIns = splito.numInstances();
        Instances split=splito.resample(new Random(1));
        double splitData[][] = new double[numIns][numAtt];
        double tmp[] = new double[numIns];
        for(int j=0; j<splitData[0].length; j++){
            tmp = split.attributeToDoubleArray(j);
            for(int k=0; k<splitData.length; k++){
                splitData[k][j] = tmp[k];
            }
        }
        return splitData;
    }
    public double[][] getDoubleFullData(){
        int numIns = data.numInstances();
        double splitData[][] = new double[numIns][numAtt];
        double tmp[] = new double[numIns];

        for(int j=0; j<splitData[0].length; j++){
            tmp = data.attributeToDoubleArray(j);
            for(int k=0; k<splitData.length; k++){
                splitData[k][j] = tmp[k];
            }
        }
        return splitData;
    }
    public Instances getArffTestCV(int i){
        return data.testCV(numSplit, i);
    }
    public Instances getArffTrainCV(int i){
        return data.trainCV(numSplit, i);
    }
    public Instances getArffFullData(){
        return data;
    }
    public int getObservationNumber(){
    	return data.numInstances();
    }    
}