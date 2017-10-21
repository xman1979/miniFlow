package com.udacity.ama.miniFlow;

public class MSE extends Node{
	private int m;
	private Value diff;

	public MSE(Node Y, Node A, String name) {
		super(new Node[] {Y, A}, "MSE-" + name);
	}

	@Override
	public void forward() {
		Node Y = inbounds.get(0);
		Node A = inbounds.get(1);
		m = Y.value.row;
		diff = Y.value.sub(A.value);
		value = diff.T().dot(diff).dot(1.0/m);
	}

	@Override
	public void backward() {
		gradients.put(inbounds.get(0), diff.dot(2.0/this.m));
		gradients.put(inbounds.get(1), diff.dot(-2.0/this.m));		
	}
}
