package com.udacity.ama.miniFlow;

public class Linear extends Node {

	public Linear(Node X, Node W, Node b, String name) {
		super(new Node[] {X, W, b}, "Linear-" + name);
	}

	@Override
	public void forward() {
		Value X = inbounds.get(0).value;
		Value W = inbounds.get(1).value;
		Value b = inbounds.get(2).value;
		this.value = X.dot(W).add(b);
	}

	@Override
	public void backward() {
		Value gradX = inbounds.get(0).value.zero();
		Value gradW = inbounds.get(1).value.zero();
		Value gradB = inbounds.get(2).value.zero();

		for (Node node : outbounds) {
			Value cost = node.gradients.get(this);
			gradX = gradX.add(cost.dot(inbounds.get(1).value.T()));
			gradW = gradW.add(inbounds.get(0).value.T().dot(cost));
			gradB = gradB.add(cost.sumRow());	
		}
		gradients.put(inbounds.get(0), gradX);
		gradients.put(inbounds.get(1), gradW);
		gradients.put(inbounds.get(2), gradB);
	}
}
