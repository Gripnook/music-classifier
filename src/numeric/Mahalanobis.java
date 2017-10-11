package numeric;

import java.util.List;

public class Mahalanobis {
	private int dataSize;
	private double[][] inverseCovariance;

	public Mahalanobis(List<double[]> data, int dataSize) {
		this.dataSize = dataSize;
		Stats stats = new Stats(dataSize);
		for (double[] datum : data) {
			stats.add(datum);
		}
		inverseCovariance = Matrix.invert(stats.covariance());
	}

	public double distance(double[] lhs, double[] rhs) {
		double result = 0;
		for (int i = 0; i < dataSize; ++i) {
			double dx = (lhs[i] - rhs[i]);
			for (int j = 0; j < dataSize; ++j) {
				double dy = (lhs[j] - rhs[j]);
				result += dx * inverseCovariance[i][j] * dy;
			}
		}
		return Math.sqrt(result);
	}
}
