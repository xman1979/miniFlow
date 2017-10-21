package com.udacity.ama.miniFlow;

public class Input extends Node{

	public Input(Value val, String name) {
		super(new Node[] {}, "Input-" + name);
		this.value = val;
	}

	public Input(String name) {
		super(new Node[] {}, "Input-" + name);
	}

	@Override
	public void forward() {
	}

	@Override
	public void backward() {
		Value total = this.value.zero();
		for (Node node : outbounds) {
			total = total.add(node.gradients.get(this));
		}
		gradients.put(this, total);
	}
}
