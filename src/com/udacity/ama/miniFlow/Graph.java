package com.udacity.ama.miniFlow;

import java.util.*;

public class Graph {
	public static List<Node> topologicalSort(Node[] inputNodes) {
		Queue<Node> nodes = new LinkedList<>();
		for (Node node : inputNodes) {
			nodes.add(node);
		}
		Map<Node, Set<Node>> inbounds = new HashMap<>();
		Map<Node, Set<Node>> outbounds = new HashMap<>();

		while(!nodes.isEmpty()) {
			Node node = nodes.poll();
			for (Node m : node.outbounds) {
				Set<Node> out = outbounds.getOrDefault(node, new HashSet<>());
				out.add(m);
				outbounds.put(node, out);
				Set<Node> in = inbounds.getOrDefault(m, new HashSet<>());
				in.add(node);
				inbounds.put(m, in);
				nodes.add(m);
			}
		}
		
		List<Node> res = new ArrayList<>();
		for (Node node : inputNodes) {
			nodes.add(node);
		}
		while(!nodes.isEmpty()) {
			Node node = nodes.poll();
			res.add(node);
			for (Node m : node.outbounds) {
				outbounds.get(node).remove(m);
				inbounds.get(m).remove(node);
				if (inbounds.get(m).isEmpty()) {
					nodes.add(m);
				}
			}
		}
		return res;
	}
	
	public static void forward(List<Node> nodes) {
		for (Node node : nodes) {
			node.forward();
			//System.out.println("After forward " + node.name + " value " + node.value.toString());
		}
	}
	
	public static void backward(List<Node> nodes) {
		for (int i = nodes.size()-1; i >= 0; i--) {
			Node node = nodes.get(i);
			node.backward();
			//for (Node g : node.inbounds) {
			//	System.out.println("After backward " + node.name + " [" + g.name + "] =" + node.gradients.get(g).toString());
			//}
		}
	}

	public static void sgdUpdate(Input[] trainables, double learningRate) {
		for (Node t : trainables) {
			Value partial = t.gradients.get(t);
			t.value = t.value.sub(partial.dot(learningRate));
		}
	}
}
