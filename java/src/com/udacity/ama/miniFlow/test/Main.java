package com.udacity.ama.miniFlow.test;

import java.util.*;
import com.udacity.ama.miniFlow.*;

public class Main {

	public static void main(String[] args) {
		int numFeatures = BostonData.FEATURES;
		int numSamples = BostonData.SAMPLES;
		int numHidden = 10;

		Value W1_ = new Value(numFeatures, numHidden).random();
		Value b1_ = new Value(1, numHidden).zero();
		Value W2_ = new Value(numHidden, 1).random();
		Value b2_ = new Value(1, 1).zero();

		// Nodes 
		Input X = new Input("X");
		Input y = new Input("Y");
		Input W1 = new Input(W1_, "W1");
		Input b1 = new Input(b1_, "b1");
		Input W2 = new Input(W2_, "W2");
		Input b2 = new Input(b2_, "b2");
	
		// Structure of the neural network
 		Node linear1 = new Linear(X, W1, b1, "l1");
		Node sigmoid = new Sigmoid(linear1, "s1");
		Node linear2 = new Linear(sigmoid, W2, b2, "l2");
		Node cost = new MSE(y, linear2, "cost");

		Node[] inputNodes = new Node[] {X, y, W1, b1, W2, b2};
		List<Node> sortedNodes = Graph.topologicalSort(inputNodes);
		for (Node node: sortedNodes) {
			System.out.println(node.name);
		}

		System.out.println("Total number of examples: " + numSamples);

		int epochs = 1000;
		int batchSize = 1;
		int stepsPerEpoch = numSamples / batchSize;
		double learningRate = 0.01;

		Input[] trainables = new Input[]{W1, b1, W2, b2};
		Value X_ = new Value(BostonData.instance.DATA).normalizeColumn();
		Value y_ = new Value(BostonData.instance.TARGET).T();
		for (int i = 0; i < epochs; i++) { 
		    double loss = 0.0;
		    for (int j = 0; j < stepsPerEpoch; j++) {
		    		int randomRowIndex = (int)(Math.random() * numSamples);
		    		X.value = X_.slice(randomRowIndex, batchSize);
		    		y.value = y_.slice(randomRowIndex, batchSize);
		    		Graph.forward(sortedNodes);
		    		Graph.backward(sortedNodes);
		    		Graph.sgdUpdate(trainables, learningRate);
		    		loss += sortedNodes.get(sortedNodes.size()-1).value.mean();
		    }
		    System.out.println("Epoch " + i + ", Loss: " + loss/stepsPerEpoch);
		}
		
		for (Node param : trainables) {
			System.out.println("Node " + param.name + " is ");
			System.out.println(param.value.toString());
		}
		
		// Validate the result of first 10 samples
		X.value = X_.slice(0, 10);
		y.value = y_.slice(0, 10);
		Graph.forward(sortedNodes);
		Value a = sortedNodes.get(sortedNodes.size()-2).value; // a is l2 output value
		System.out.println("Calculate result " + a.T().toString());
		System.out.println("Expected result " + y.value.T().toString());
	}
}
