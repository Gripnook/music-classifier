package numeric;

import org.ejml.simple.SimpleMatrix;

public class Matrix {
	public static SimpleMatrix convertMatrix(double[][] matrix) {
		SimpleMatrix result = new SimpleMatrix(matrix.length, matrix.length);
		for (int i = 0; i < matrix.length; ++i) {
			if (matrix[i].length != matrix.length)
				throw new Error("matrix must be square");
			for (int j = 0; j < matrix.length; ++j) {
				result.set(i, j, matrix[i][j]);
			}
		}
		return result;
	}

	public static double[][] convertMatrix(SimpleMatrix matrix) {
		double[][] result = new double[matrix.numRows()][matrix.numCols()];
		for (int i = 0; i < matrix.numRows(); ++i) {
			for (int j = 0; j < matrix.numCols(); ++j) {
				result[i][j] = matrix.get(i, j);
			}
		}
		return result;
	}

	public static SimpleMatrix convertVector(double[] vector) {
		SimpleMatrix result = new SimpleMatrix(vector.length, 1);
		for (int i = 0; i < vector.length; ++i) {
			result.set(i, vector[i]);
		}
		return result;
	}

	public static double[] convertVector(SimpleMatrix vector) {
		if (!vector.isVector())
			throw new Error("matrix must be a vector");
		double[] result = new double[vector.getNumElements()];
		for (int i = 0; i < vector.getNumElements(); ++i) {
			result[i] = vector.get(i);
		}
		return result;
	}

	public static double[][] invert(double[][] matrix) {
		return convertMatrix(convertMatrix(matrix).invert());
	}

	private Matrix() {

	}
}
