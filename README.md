Neural Language Model Exercise
=========================================

This is a Java skeleton code for the neural language model exercise, created for the Machine Learning for Language Modelling course. More information on the course homepage:
http://www.marekrei.com/teaching/mllm/

The task
-----------------------------------------

The parts you need to fill in are in the Network.java file, marked by the "TODO" markers. However, you are free to modify other parts of the file as you wish and you may want to create new functions to separate out certain tasks.

The **feedForward** function gets integer ids of the context words and the correct next word as input. The system needs to calculate word probabilities into the output vector and return the log probability (in base 10) of the correct word.

The **backProp** function gets the same ids as input, plus the alpha value (learning rate). Using the output vector the system calculated in the feedForward function, this function now needs to calculate error derivatives across the network and update the weights for W, Wout and E.

Do not put your code publically online before the end of the homework deadline.

Running the system
-----------------------------------------

First you'll need to compile the Java code. This process will depend on your operating system, but on Linux/Mac this command should work:

        javac -d bin -sourcepath src -cp lib/* src/neurallm/*

Or you can install and use eclipse, which will handle linking and compiling

        [https://www.eclipse.org](https://www.eclipse.org)

I have implemented the gradient check, as described in the lectures, to test your code. You can run

	java -cp bin:lib/* neurallm.Network

to perform the gradient check. If you have implemented the feedforward part correctly and the test passes, then that indicates that the backpropagation is correct as well.

In order to run the training and scoring, run

	java -cp bin:lib/* neurallm.LM trainingfile devfile scoringinput scoringoutput

**scoringinput** is the file you want to add scores to, such as the input file from the [language modelling task](http://www.marekrei.com/teaching/lmtask). **scoringoutput** is the output file name where scores will be written.

The system will train on the **trainingdata**, testing on the **devdata** every 4000 lines by default. If the performance has not improved, the learning rate will be decreased at every epoch. When there is again no improvement, the training will stop.

There are some settings in the beginning of LM.java that you can change if you wish.

This implementation is not the most optimal way of implementing neural language models. Many advanced techniques exist for speeding up such models, and you are encouraged to find out about them if you are interested. This code is kept simple and minimal to introduce the concepts of neural language modelling and parameter training through backpropagation.



JBlas
-----------------------------------------

This code uses [JBlas](http://jblas.org/), a linear algebra library for Java. We are using the DoubleMatrix class, and various operations can be called on it, such as addition, multiplication, etc. You can look at the documentation for details and a useful introduction can be found here:
http://jblas.org/javadoc/org/jblas/DoubleMatrix.html


