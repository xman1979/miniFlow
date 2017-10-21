package com.udacity.ama.miniFlow;
import java.util.*;

public abstract class Node {
	public String name;
	public Value value;
	List<Node> inbounds;
	List<Node> outbounds;
	Map<Node, Value> gradients;

	public Node(Node[] inputs, String name) {
		this.name = name;
		this.inbounds = new ArrayList<>();
		this.outbounds = new ArrayList<>();
		for (Node node : inputs) {
			this.inbounds.add(node);
			node.outbounds.add(this);
		}
		this.gradients = new HashMap<>();
	}
	
	public abstract void forward();
	public abstract void backward();	
}
