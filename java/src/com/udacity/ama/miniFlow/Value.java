package com.udacity.ama.miniFlow;

public class Value {

	public double[][] M;
	public int row;
	public int col;
	
	public Value(double[][] val) {
		this.row = val.length;
		this.col = val[0].length;
		this.M = new double[row][col];
		fill(val);
	}
	
	public Value(double[] val) {
		this.row = 1;
		this.col = val.length;
		this.M = new double[row][col];
		for (int i = 0; i < val.length; i++) {
			M[0][i] = val[i];
		}
	}

	public Value(int row, int col) {
		this.M = new double[row][col];
		this.row = row;
		this.col = col;
	}
	
	public Value random() {
		Value V = new Value(row, col);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				V.M[i][j] = Math.random();
			}
		}
		return V;
	}

	public void fill(double val) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				M[i][j] = val;
			}
		}
	}

	public void fill(double[][] val) {
		for (int i = 0; i < val.length; i++) {
			for (int j = 0; j < val[0].length; j++) {
				M[i][j] = val[i][j];
			}
		}
	}
	
	public Value add(Value V) {
		if (row == V.row && col == V.col) {
			double[][] res = new double[row][col];
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					res[i][j] = M[i][j] + V.M[i][j];
				}
			}
			return new Value(res);
		}
		
		if (V.row == 1 && col == V.col) {
			double[][] res = new double[row][col];
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					res[i][j] = M[i][j] + V.M[0][j];
				}
			}
			return new Value(res);			
		}
		return null;
	}
	
	public Value sumRow() {
		double[] res = new double[col];
		for (int j = 0; j < col; j++) {
			res[j] = 0.0; 
			for (int i = 0; i < row; i++) {
				res[j] += M[i][j];
			}
		}
		return new Value(res);
	}

	public Value sub(Value V) {
		if (row == V.row && col == V.col) {
			double[][] res = new double[row][col];
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					res[i][j] = M[i][j] - V.M[i][j];
				}
			}
			return new Value(res);
		}		
		if (V.row == 1 && col == V.col) {
			double[][] res = new double[row][col];
			for (int i = 0; i < row; i++) {
				for (int j = 0; j < col; j++) {
					res[i][j] = M[i][j] - V.M[0][j];
				}
			}
			return new Value(res);			
		}
		return null;
	}
	
	public Value subFromEachRow(Value V) {
		if (V.row != 1 || col != V.col) {
			return null;
		}
		double[][] res = new double[row][col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				res[i][j] = M[i][j] - V.M[0][j];
			}
		}
		return new Value(res);
	}

	public Value T() {
		double[][] res = new double[col][row];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				res[j][i] = M[i][j];
			}
		}
		return new Value(res);
	}
	
	public Value sigmoid() {
		double[][] res = new double[row][col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				res[i][j] = 1.0 / (1.0 + Math.exp(0.0 - M[i][j]));
			}
		}
		return new Value(res);
	}

	public Value sigmoidPrime() {
		double[][] res = new double[row][col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				double sigmoid = 1.0 / (1.0 + Math.exp(0.0 - M[i][j]));
				res[i][j] = sigmoid * (1 - sigmoid);
			}
		}
		return new Value(res);
	}
	
	public Value dot(Value V) {
		// [ row, col ] * [V.row, V.col] 
		if (col != V.row) {
			return null;
		}
		double[][] res = new double[row][V.col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < V.col; j++) {
				res[i][j] = 0.0;
				for (int k = 0; k < col; k++) {
					res[i][j] += M[i][k] * V.M[k][j];
				}
			}
		}
		return new Value(res);
	}

	public Value multiply(Value V) {
		if (row != V.row || col != V.col) {
			return null;
		}
		double[][] res = new double[row][col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < V.col; j++) {
				res[i][j] = M[i][j] * V.M[i][j];
			}
		}
		return new Value(res);
	}

	public Value dot(double v) {
		double[][] res = new double[row][col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				res[i][j] = M[i][j] * v;
			}
		}
		return new Value(res);
	}
	
	public double sum() {
		double res = 0.0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				res += M[i][j];
			}
		}
		return res;
	}

	public double mean() {
		double res = 0.0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				res += M[i][j];
			}
		}
		return res/(row * col);
	}
	
	public double std() {
		double mean = mean();
		double res = 0.0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				res += (M[i][j] - mean) * (M[i][j] - mean);
			}
		}
		return Math.sqrt(res/(row * col));
	}

	public Value normalize() {
		double[][] res = new double[row][col];
		double mean = mean();
		double std = std();
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				res[i][j] = (M[i][j] - mean)/std;
			}
		}
		return new Value(res);		
	}

	public Value normalizeColumn() {
		double[][] res = new double[row][col];
		Value t = T();
		for (int i = 0; i < t.row; i++) {
			Value c = new Value(t.M[i]).normalize();
			for (int j = 0; j < t.col; j++) {
				res[j][i] = c.M[0][j];
			}
		}
		return new Value(res);
	}
	
	public Value zero() {
		double[][] res = new double[row][col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				res[i][j] = 0.0;
			}
		}
		return new Value(res);		
	}
	
	public Value slice(int start, int batchSize) {
		if (batchSize >= row) {
			return new Value(M);
		}
		double[][] res = new double[batchSize][col];
		for (int i = 0; i < batchSize; i++) {
			int index = (start + i) % row;
			for (int j = 0; j < col; j++) {
				res[i][j] = M[index][j];
			}
		}
		return new Value(res);
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("(" + row + ", " + col + ")\n");
		for (int i = 0; i < row; i++) {
			sb.append("[");
			for (int j = 0; j < col; j++) {
				sb.append(String.format("%.2f", M[i][j]));
				sb.append(" ");
			}
			sb.append("]\n");
		}
		return sb.toString();
	}

}
