package com.udacity.ama.miniFlow;

public class Sigmoid extends Node {

	public Sigmoid(Node input, String name) {
		super(new Node[] {input}, "Sigmoid-" + name);
	}

	@Override
	public void forward() {
		Node input = inbounds.get(0);
		value = input.value.sigmoid();
	}

	@Override
	public void backward() {
		Value grads = inbounds.get(0).value.zero();
		for (Node node : outbounds) {
			Value cost = node.gradients.get(this);
			grads = grads.add(inbounds.get(0).value.sigmoidPrime().multiply(cost));
		}
		this.gradients.put(inbounds.get(0), grads);
	}
}
