package neurallm;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.LinkedList;
import java.util.List;

/**
 * A class for creating and running a neural language model.
 *
 */
public class LM {
	public int N = 3; // N-gram size. N-1 words will be used as context.
	public int M = 30; // Size of the word representations
	public int H = 30; // Size of the hidden layer
	public int epochLines = 4000; // Number of lines processed at every epoch
	public double alpha = 0.1; // Initial learning rate

	public String unk = "<UNK>";
	public String start = "<S>";
	public String end = "</S>";
	
	Dictionary dict;
	Network network;
	
	boolean alphaDivide = false;
	
	public LM(String trainingFile){
		this.createDictionary(trainingFile);
		this.network = new Network(N, M, H, this.dict.size());
	}
	
	/**
	 * Creates the dictionary from a training file
	 * 
	 * @param trainingFile The input training file to process
	 */
	public void createDictionary(String trainingFile){
		System.out.println("Creating the dictionary");
		dict = new Dictionary();
		try (BufferedReader br = new BufferedReader(new FileReader(trainingFile))) {
			String line;
			while ((line = br.readLine()) != null){
				String[] words = line.trim().split("\\s+");
				for(String word : words)
					dict.add(word);
			}
			dict.add(unk);
			dict.add(start);
			dict.add(end);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Reads in text line by line writes the average log probability
	 * to the output file.
	 * 
	 * @param scoringInput The input file, containing one sentence per line
	 * @param scoringOutput Output file where the scores will be written
	 */
	public void runScoring(String scoringInput, String scoringOutput){
		System.out.println("Scoring " + scoringInput);
		try{
			BufferedReader br = new BufferedReader(new FileReader(scoringInput));
			PrintWriter writer = new PrintWriter(scoringOutput, "UTF-8");
			
			String line;
			while ((line = br.readLine()) != null){
				for(int i = 1; i < N; i++)
					line = start + " " + line;
				line = line + " " + end;
				String[] words = line.trim().split("\\s+");
				
				List<Integer> context = new LinkedList<Integer>();
				for(int i = 0; i < N-1; i++)
					context.add(getWordId(words[i]));

				double logp = 0.0;
				int wordCount = 0;
				for(int i = N-1; i < words.length; i++){
					int id = getWordId(words[i]);
					logp += network.feedForward(context, id);
					wordCount++;
					
					context.remove(0);
					context.add(id); 
				}
				double logpAvg = (1.0 / (double)wordCount) * logp;
				writer.println(logpAvg + "");
			}
			
			br.close();
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Get the integer id of a word from the dictionary
	 * 
	 * @param word
	 * @return id
	 */
	public int getWordId(String word){
		int id = dict.getId(word);
		if(id < 0)
			id = dict.getId(unk);
		return id;
	}
	
	/**
	 * Processes an input file and returns the perplexity of the text.
	 * If alpha is greater than zero, the network backpropagation is used for training.
	 * If the devFile is set, function is called recursively at certain intervals
	 * to measure performance on the development set. If the performance does not improve,
	 * the learning rate is decreased or the learning is stopped.
	 * 
	 * @param file The main text file being processed
	 * @param devFile The file to use for evaluation
	 * @param alpha Learning rate
	 * @return
	 */
	public double processFile(String file, String devFile, double alpha){
		double logp = 0.0;
		int wordCount = 0;
		int lineCount = 0;
		double pplDevOld = Double.MAX_VALUE;
		int epoch = 0;
		
		if(alpha > 0.0)
			System.out.println("Epoch " + epoch);

		try{
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;
			while ((line = br.readLine()) != null){
				for(int i = 1; i < N; i++)
					line = start + " " + line;
				line = line + " " + end;
				String[] words = line.trim().split("\\s+");
				
				List<Integer> context = new LinkedList<Integer>();
				for(int i = 0; i < N-1; i++)
					context.add(getWordId(words[i]));

				for(int i = N-1; i < words.length; i++){
					int id = getWordId(words[i]);
					logp += network.feedForward(context, id);
					wordCount++;
					
					if(alpha > 0.0)
						network.backProp(context, id, alpha);
					
					context.remove(0);
					context.add(id); 
				}
				
				if(devFile != null && ++lineCount % epochLines == 0){
					double pplTrain = Math.pow(10.0, (-1.0 / (double)wordCount) * logp);
					System.out.println("PPL_train: " + pplTrain);
					
					double pplDev = processFile(devFile, null, 0.0);
					System.out.println("PPL_dev: " + pplDev);
					
					if (-1.0 * Math.log10(pplDev) * 1.003 < -1.0 * Math.log10(pplDevOld)) {
						if (!this.alphaDivide)
							this.alphaDivide = true;
						else
							break;
					}
					if (this.alphaDivide)
						alpha /= 2.0;
					pplDevOld = pplDev;

					System.out.println("Epoch " + (++epoch) + " ( alpha = " + alpha + ")");
				}
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		double ppl = Math.pow(10.0, (-1.0 / (double)wordCount) * logp);
		return ppl;
	}
	
	public static void main(String[] args){
		String trainingFile = args[0];
		String devFile = args[1];
		String scoringInput = args[2];
		String scoringOutput = args[3];
		
		LM lm = new LM(trainingFile);
		lm.processFile(trainingFile, devFile, lm.alpha);
		lm.runScoring(scoringInput, scoringOutput);
	}
}
