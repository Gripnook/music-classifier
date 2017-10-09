public class Matrix {
	public static double[][] invert(double[][] matrix) {
		int n = matrix.length;
		for (int i = 0; i < n; ++i) {
			if (matrix[i].length != n) {
				throw new Error("matrix must be square");
			}
		}

		double[][] augment = augment(matrix);

		// Forward elimination.
		for (int i = 0; i < n; ++i) {
			pivot(augment, i);
			double pivot = augment[i][i];
			if (pivot == 0)
				return null;
			for (int j = i; j < 2 * n; ++j) {
				augment[i][j] /= pivot;
			}
			for (int j = i + 1; j < n; ++j) {
				double factor = augment[j][i];
				for (int k = i; k < 2 * n; ++k) {
					augment[j][k] -= factor * augment[i][k];
				}
			}
		}

		// Back substitution.
		for (int i = n - 1; i >= 0; --i) {
			for (int j = i - 1; j >= 0; --j) {
				double factor = augment[j][i];
				for (int k = i; k < 2 * n; ++k) {
					augment[j][k] -= factor * augment[i][k];
				}
			}
		}

		double[][] result = new double[n][n];
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				result[i][j] = augment[i][j + n];
			}
		}
		return result;
	}

	private static double[][] augment(double[][] matrix) {
		int n = matrix.length;
		double[][] augment = new double[n][2 * n];
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				augment[i][j] = matrix[i][j];
			}
		}
		for (int i = 0; i < n; ++i) {
			augment[i][i + n] = 1;
		}
		return augment;
	}

	private static void pivot(double[][] matrix, int i) {
		int pivotRow = i;
		double pivot = Math.abs(matrix[i][i]);
		for (int j = i + 1; j < matrix.length; ++j) {
			if (Math.abs(matrix[j][i]) > pivot) {
				pivotRow = j;
				pivot = Math.abs(matrix[j][i]);
			}
		}
		double[] temp = matrix[i];
		matrix[i] = matrix[pivotRow];
		matrix[pivotRow] = temp;
	}

	private Matrix() {

	}
}
