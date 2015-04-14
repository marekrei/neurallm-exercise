package neurallm;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.jblas.DoubleMatrix;

/**
 * A neural network language model
 *
 */
public class Network {
	Map<Integer,DoubleMatrix> E; // Contains vectors of word representations
	DoubleMatrix[] W; // N-1 weight matrices for input->hidden step
	DoubleMatrix Wout; // Weight matrix for hidden->output step
	
	DoubleMatrix hidden; // Hidden vector
	DoubleMatrix output; // Output vector
	
	// Error vectors/matrices
	DoubleMatrix errorOutput;
	DoubleMatrix errorWout;
	DoubleMatrix errorHidden;
	DoubleMatrix[] errorW;	
	DoubleMatrix[] errorInput;
	
	Random random = new Random(1);
	
	/** 
	 * Create a new neural language model with N-1 context words, 
	 * M size word representations, H size hidden vector,
	 * and V size vocabulary size
	 * 
	 * @param N N-gram size
	 * @param M Word representation size
	 * @param H Hidden vector size
	 * @param V Vocabulary size
	 */
	
	public Network(int N, int M, int H, int V){
		E = new LinkedHashMap<Integer,DoubleMatrix>();
		for(int i = 0; i < V; i++){
			DoubleMatrix v = DoubleMatrix.zeros(M, 1);
			initMatrixRandom(v);
			E.put(i, v);
		}
		
		W = new DoubleMatrix[N-1];
		for(int i = 0; i < W.length; i++){
			W[i] = DoubleMatrix.zeros(H, M);
			initMatrixRandom(W[i]);
		}
		
		Wout = DoubleMatrix.zeros(V, H);
		initMatrixRandom(Wout);
		
		hidden = DoubleMatrix.zeros(H, 1);
		output = DoubleMatrix.zeros(V, 1);
		
		
		errorOutput = DoubleMatrix.zeros(V, 1);
		errorWout = DoubleMatrix.zeros(V, H);
		errorHidden = DoubleMatrix.zeros(H, 1);
		errorW = new DoubleMatrix[N-1];
		for(int i = 0; i < errorW.length; i++)
			errorW[i] = DoubleMatrix.zeros(W[i].rows, W[i].columns);
		errorInput = new DoubleMatrix[N-1];
		for(int i = 0; i < errorW.length; i++)
			errorInput[i] = DoubleMatrix.zeros(W[i].columns, 1);
		
	}
	
	/**
	 * Initialise the matrix with random normally distributed values
	 * 
	 * @param m
	 */
	public void initMatrixRandom(DoubleMatrix m){
		for(int i = 0; i < m.rows; i++)
			for(int j = 0; j < m.columns; j++)
				m.put(i, j, randomValue());
	}
	
	/**
	 * A helper function for generating a normally distributed random value
	 * 
	 * @return
	 */
	public double randomValue(){
		return _randomValue() + _randomValue() + _randomValue();
	}
	
	public double _randomValue(){
		return (random.nextDouble() * 0.2) - 0.1;
	}
	
	/**
	 * Calculates the output probabilities of all words, and 
	 * returns the log probability (in base 10) of the correct word.
	 * 
	 * @param context List of integers for the context words
	 * @param nextWord Integer id for the next word
	 * @return the log probability (in base 10) of the correct word
	 */
	public double feedForward(List<Integer> context, Integer nextWord){
		// TODO: implement the feedforward activation process
		return -10000.0;
	}
	
	/**
	 *  Using the output vector the system calculated in the feedForward function, 
	 *  this function now needs to calculate error derivatives across the network 
	 *  and update the weights for W, Wout and E.
	 *  
	 * @param context List of integers for the context words
	 * @param nextWord Integer id for the next word
	 * @param alpha Learning rate
	 */
	public void backProp(List<Integer> context, Integer nextWord, double alpha){
		// TODO: implement backpropagation
	}
	
	/**
	 * Perform the gradient check on the backprop implementation
	 */
	public static void gradientCheck(){
		List<Integer> context = Arrays.asList(1, 2, 3);
		Integer nextWord = 5;
		Network network = new Network(4, 20, 30, 40);
		
		double epsilon = 0.0001;
		double alpha = 0.1;
		
		int i = 1, j = 12;
		double originalValue = network.E.get(i).get(j, 0);
		
		network.E.get(i).put(j, 0, originalValue + epsilon);
		network.feedForward(context, nextWord);
		double loss1 = -1.0 * Math.log(network.output.get(nextWord, 0));
		
		network.E.get(i).put(j, 0, originalValue - epsilon);
		network.feedForward(context, nextWord);
		double loss2 = -1.0 * Math.log(network.output.get(nextWord, 0));
		
		double derivative1 = (loss1 - loss2)/(2.0 * epsilon);
		System.out.println("Derivative1: " + derivative1);
		
		network.E.get(i).put(j, 0, originalValue);
		network.feedForward(context, nextWord);
		network.backProp(context, nextWord, alpha);
		double derivative2 = (network.E.get(i).get(j, 0) - originalValue) / (-1.0 * alpha);
		System.out.println("Derivative2: " + derivative2);
		
		if(Math.abs(derivative1 - derivative2) < 0.00001)
			System.out.println("Gradient check: PASS");
		else
			System.out.println("Gradient check: FAIL");
	}
	
	public static void main(String[] args){
		Network.gradientCheck();
	}
}
